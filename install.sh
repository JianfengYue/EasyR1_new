# module load cuda/12.6
# module load gcc/11

# 克隆源码并安装 flash-attn
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.8.2
export TMPDIR=$PWD/tmp
mkdir -p $TMPDIR
pip install --no-build-isolation --no-binary flash-attn .