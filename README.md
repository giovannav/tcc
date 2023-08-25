# tcc

## **run as:**
#### python vgg16.py --model vgg16 --output_nodes 256 --epochs 1 --img_size 128 --batch_size 8 --learning_rate 0.001
#### (tegrastats --interval 60000 2>&1 | while IFS= read -r line; do echo "$(date +'%Y-%m-%d %H:%M:%S') $line"; done) >> vgg16.log &