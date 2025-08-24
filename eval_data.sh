project_name='sig_amc'
prefix='/path/to/your/dataset/'

# you can customize the tx_pwr / distance / sample_rate / and inter

python eval_only.py --prefix $prefix --tx_pwr="['TP-10',]" --distance="['D5','D10','D15']" --sample_rate="['SR20',]" \
--inter="['I0','I1']" 