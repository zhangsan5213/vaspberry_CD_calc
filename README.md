## vaspberry计算晶体CD

1，vasp，先优化几何，再计算出WAVECAR和BAND.dat（能带计算必须要SOC）。

2，执行“vaspberry -kx 1 -ky 1 -cd 1”可以得到VBM、CBM的序号，比如示例数据里是240~241。

3，计算 VBM-5 ~ VBM 到 CBM ~ CBM+5 之间的25个跃迁（可以加多，更精确）（可以bash做，参照batch_CD.sh）。

4，把能带i、j间跃迁的CIRC_DICHROISM.dat的kx、ky、kz，对应到BAND_R_SOC.dat中能带i、j在kx、ky、kz处的能量差，即将selectivity对应到了E。

5，将所有跃迁的 selectivity-E 关系叠加。
