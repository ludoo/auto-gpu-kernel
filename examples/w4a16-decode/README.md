# w4a16-decode — the W4A16 linear's decode end (sparkle task next.120, round 2)

The second round on [w4a16-gemm](../w4a16-gemm/README.md): the start is that round's survivor (DeepSeek flash, 1.1175 on 18 cells), whose chunk cells are kept and whose decode cells sit at 1.10–1.22 of the wheel's Marlin where Marlin streams the weight at the DRAM ceiling. Same contract, same tests, same six shapes; the bench scores the M = 1 and M = 4 cells only and the prefill tile is frozen. Everything else as in the first round's README.
