
regieme="--trainer split --lr 0.01 --batch-size 64 --memory-budget 1000"

for i in {0..9}
do
    echo "Experiment $i"
    python3 -u main.py --seed $i --dataset FMNIST   $regieme --base-classes 2  --step-size 2  --rho 1  --nepochs 7   --name S_FMNIST_$i   > logs/S_FMNIST_$i.txt
    python3 -u main.py --seed $i --dataset CIFAR10  $regieme --base-classes 2  --step-size 2  --rho 1  --nepochs 17  --name S_CIFAR10_$i  > logs/S_CIFAR10_$i.txt
    python3 -u main.py --seed $i --dataset CIFAR100 $regieme --base-classes 10 --step-size 10 --rho 1  --nepochs 33 --name S_CIFAR100_$i  > logs/S_CIFAR100_$i.txt
   python3  -u main.py --seed $i --dataset CORE50   $regieme --base-classes 5  --step-size 5  --rho 1  --nepochs 2   --name S_CORE50_$i  > logs/S_CORE50_$i.txt
done

