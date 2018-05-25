wget 'https://www.dropbox.com/s/p7ssg3asyw3m2kf/vae_epoch_9.pt?dl=1' -O 'vae_epoch_9.pt'

python3 vae_random.py $1 $2
python3 vae_test.py  $1 $2
python3 dcgan_random.py $1 $2
python3 acgan_random.py $1 $2
python3 vae_tsne.py $1 $2
