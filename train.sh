source /usr/local/bin/virtualenvwrapper.sh
workon pytorch2.0.1

echo loacte
python train/train_loacte_25.py
python train/train_loacte_1.py

echo spine_segment
python train/train_spine_segment.py

echo vertbral_segment
python train/train_vertbral_segment.py