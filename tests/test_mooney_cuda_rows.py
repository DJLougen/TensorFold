"""The CUDA Mooney caller must pass live rows, not reusable-buffer capacity."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("rows", [1, 3, 8])
def test_mooney_moe_slices_reusable_buffers(monkeypatch, rows):
    from tensorfold.families.qwen4_exp.cuda import forward as module

    capacity, slots, hidden, width = 8, 3, 4, 4
    sentinel = -19.0
    buf = SimpleNamespace(
        logits=torch.full((capacity, 2), sentinel),
        act=torch.full((capacity, slots, width), sentinel, dtype=torch.bfloat16),
        y=torch.full((capacity, slots, hidden), sentinel, dtype=torch.bfloat16),
        rot_g=torch.full((capacity, hidden), sentinel, dtype=torch.bfloat16),
        rot_u=torch.full((capacity, hidden), sentinel, dtype=torch.bfloat16),
        rot_d=torch.full((capacity, slots, width), sentinel, dtype=torch.bfloat16),
        plan=SimpleNamespace(tile=0), slots=slots,
    )
    x = torch.ones((rows, hidden), dtype=torch.bfloat16)

    def router(inp, weights, out):
        assert inp.shape[0] == out.shape[0] == rows
        out.fill_(1)

    def select(logits, buffers, top_k, experts, tile):
        assert logits.shape[0] == rows

    class Rotation:
        def apply(self, inp, out):
            assert inp.shape == out.shape
            out.copy_(inp)

    def gate_up(gate, up, plan, out):
        assert gate.shape == up.shape == (rows, hidden)
        assert out.shape == (rows * slots, width)
        out.fill_(2)

    def down(inp, plan, out):
        assert inp.shape == (rows * slots, width)
        assert out.shape == (rows * slots, hidden)
        out.fill_(3)

    def shared(inp):
        assert inp.shape == (rows, hidden)
        return torch.ones((rows, width), dtype=torch.bfloat16)

    def shared_down(inp, *, out):
        assert inp.shape == (rows, width)
        assert out.shape == (rows, hidden)
        out.fill_(4)

    monkeypatch.setattr(module.moe_mod, "router", router)
    monkeypatch.setattr(module.moe_mod, "select", select)
    ex = SimpleNamespace(rot_gate=Rotation(), rot_up=Rotation(), rot_down=Rotation(),
                         gate_up=gate_up, down_proj=down, shared_gate=shared,
                         shared_up=shared, shared_down=shared_down, shared_width=width)
    assert module._mooney_moe(x, ex, buf, 2, 2, torch.ones((2, hidden))) is buf
    assert torch.all(buf.y[:rows, :2] == 3)
    assert torch.all(buf.y[:rows, 2] == 4)
    for value in (buf.logits, buf.act, buf.y, buf.rot_g, buf.rot_u, buf.rot_d):
        assert torch.all(value[rows:] == sentinel)
