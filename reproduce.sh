# FC-siamese
python main.py --config=CD_siamunet_conc_2d --eval
python main.py --config=CD_siamunet_conc_3d --eval
python main.py --config=CD_siamunet_conc --eval
python main.py --config=CD_siamunet_conc_plabel --eval

# SNUNet
python main.py --config=CD_snunet_2d --eval
python main.py --config=CD_snunet_3d --eval
python main.py --config=CD_snunet --eval


# changeformer
python main.py --config=CD_changeformer_2d --eval
python main.py --config=CD_changeformer_3d --eval
python main.py --config=CD_changeformer --eval
python main.py --config=CD_changeformer_plabel --eval

# P2VNet
python main.py --config=CD_p2vnet_2d --eval
python main.py --config=CD_p2vnet_3d --eval
python main.py --config=CD_p2vnet --eval

# MTBIT
python main.py --config=CD_MTBIT_2d --eval
python main.py --config=CD_MTBIT_3d --eval
python main.py --config=CD_MTBIT --eval
python main.py --config=CD_MTBIT_plabel --eval

# Ablation study
python main.py --config=CD_mmcdnet_2d --eval
python main.py --config=CD_mmcdnet_3d --eval
python main.py --config=CD_mmcdnet --eval
python main.py --config=CD_mmcdnet_plabel --eval


