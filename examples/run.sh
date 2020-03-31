CUDA_VISIBLE_DEVICES=0 nvprof --csv --print-api-trace  --print-gpu-trace --print-nvlink-topology --print-pci-topology \
  -o "reddit_%h_%p.prof" \
  python ./reddit.py 2> xx.log |tee xx



#CUDA_VISIBLE_DEVICES=0  \
#  python ./reddit.py 
