
# remove leftover data set
rm -rf content/train/*
rm -rf content/val/*
rm words.txt

python gen.py
# gen train data
trdg --output_dir content/train -c 10000 -l en -rs -num -let  -w 1 --case upper  -ft fonts/licenseplate.ttf -t 3 -k 20 -rk -bl 1 -rbl -na 0 --image_dir content/bkground --background 3
trdg --output_dir content/train -c 10000 -l en -rs -num -let  -w 1 --case upper  -ft fonts/MANDATOR.ttf -t 3 -k 20 -rk -bl 1 -rbl -na 0 --image_dir content/bkground --background 3
trdg --output_dir content/train -c 10000 -l en -rs -num -let  -w 1 --case upper  -ft fonts/Soxe2banh.ttf -t 3 -k 20 -rk -bl 1 -rbl -na 0 --image_dir content/bkground --background 3
trdg --output_dir content/train -c 10000 -i words.txt  -w 1 --case upper  -ft fonts/Soxe2banh.ttf -t 3 -k 20 -rk -bl 1 -rbl -na 0 --image_dir content/bkground --background 3
trdg --output_dir content/train -c 10000 -i words.txt  -w 1 --case upper  -ft fonts/MANDATOR.ttf -t 3 -k 20 -rk -bl 1 -rbl -na 0 --image_dir content/bkground --background 3
trdg --output_dir content/train -c 10000 -i words.txt  -w 1 --case upper  -ft fonts/licenseplate.ttf -t 3 -k 20 -rk -bl 1 -rbl -na 0 --image_dir content/bkground --background 3


# gen val data
trdg --output_dir content/val -c 1000 -l en -rs -num -let -w 1 --case upper  -ft fonts/Soxe2banh.ttf -t 3 -k 20 -rk -bl 1 -rbl -na 0 --image_dir content/bkground --background 3
trdg --output_dir content/val -c 1000 -l en -rs -num -let -w 1 --case upper  -ft fonts/MANDATOR.ttf -t 3 -k 20 -rk -bl 1 -rbl -na 0 --image_dir content/bkground --background 3
trdg --output_dir content/val -c 1000 -l en -rs -num -let -w 1 --case upper  -ft fonts/licenseplate.ttf -t 3 -k 20 -rk -bl 1 -rbl -na 0 --image_dir content/bkground --background 3
trdg --output_dir content/val -c 1000 -i words.txt  -w 1 --case upper  -ft fonts/licenseplate.ttf -t 3 -k 20 -rk -bl 1 -rbl -na 0 --image_dir content/bkground --background 3
trdg --output_dir content/val -c 1000 -i words.txt  -w 1 --case upper  -ft fonts/MANDATOR.ttf  -t 3 -k 20 -rk -bl 1 -rbl -na 0 --image_dir content/bkground --background 3
trdg --output_dir content/val -c 1000 -i words.txt  -w 1 --case upper  -ft fonts/Soxe2banh.ttf -t 3 -k 20 -rk -bl 1 -rbl -na 0 --image_dir content/bkground --background 3