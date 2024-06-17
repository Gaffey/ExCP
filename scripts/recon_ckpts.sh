model_path=$1/models
for iter in {2000..35000..1000}
do
    last_iter=$[iter-1000]
    python compress_pythia.py $model_path/checkpoint-$iter $model_path/checkpoint-$last_iter --output $model_path/checkpoint-$iter --only_recon
done
python compress_pythia.py $1/models2/checkpoint-1000 $1/models/checkpoint-35000 --output $1/models2/checkpoint-1000 --only_recon
model_path=$1/models2
for iter in {2000..23000..1000}
do
    last_iter=$[iter-1000]
    python compress_pythia.py $model_path/checkpoint-$iter $model_path/checkpoint-$last_iter --output $model_path/checkpoint-$iter --only_recon
done
python compress_pythia.py $model_path/checkpoint-23886 $model_path/checkpoint-23000 --output $model_path/checkpoint-23886 --only_recon

