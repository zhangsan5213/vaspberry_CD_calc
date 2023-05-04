## vaspberry计算晶体CD

1，必须要SOC，先vasp优化、计算出WAVECAR和BAND.dat。

2，执行“vaspberry -kx 1 -ky 1 -cd 1”可以得到VBM、CBM的序号，比如下面例子里是240~241。

3，计算上下个五个之间的跃迁，可以bash做，参照batch_CD.sh。

4，把CIRC_DICHROISM.dat里面的kxkykz对应到K空间里面的kxkykz，对应到该两条带之间的跃迁能量。

5，将25个跃迁的selectivity加和
