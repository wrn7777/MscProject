from matplotlib import pyplot as plt
import pandas as pd

widths = ['0.2','0.45','0.7','1.0']
# ys = ['loss','prec1','prec5']
ys = ['loss','prec1']
styles = ['-o','-v','-p','-*']

shor_path = '/Users/ryan/GitHub/MscProject/results/results_pretrained_16/'
scra_path = '/Users/ryan/GitHub/MscProject/results/results_scratch/'
base_path = '/Users/ryan/GitHub/MscProject/results/results_pretrained/'

model_32_name = 'egogesture_mobilenetv2_{0:s}x_RGB_32_train.log'
model_16_name = 'egogesture_mobilenetv2_{0:s}x_RGB_16_train.log'

# 各组横向对比横向对比

for y in ys:
    for width,style in zip(widths,styles):
        df_pr = pd.read_csv(base_path + model_32_name.format(width), delimiter='\t')
        plt.plot(range(len(df_pr)),df_pr[y],style,label=width)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel(y)
    plt.title('width multipliers comparison with 32 sample duration')
    plt.savefig('/Users/ryan/GitHub/MscProject/analysis/vis_32/'+y+'_widths_pretrained.png',dpi=400)
    plt.cla()
    plt.close()

for y in ys:
    for width,style in zip(widths,styles):
        df_sc =pd.read_csv(scra_path + model_32_name.format(width), delimiter='\t')
        plt.plot(range(len(df_sc)),df_sc[y],style,label=width)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel(y)
    plt.title('width multipliers comparison with trained from scratch')
    plt.savefig('/Users/ryan/GitHub/MscProject/analysis/vis_32/'+y+'_widths_scratch.png',dpi=400)
    plt.cla()
    plt.close()

for y in ys:
    for width,style in zip(widths,styles):
        df_sc =pd.read_csv(shor_path + model_16_name.format(width), delimiter='\t')
        plt.plot(range(len(df_sc)),df_sc[y],style,label=width)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel(y)
    plt.title('width multipliers comparison with models 16 sample duration')
    plt.savefig('/Users/ryan/GitHub/MscProject/analysis/vis_32/'+y+'_widths_16.png',dpi=400)
    plt.cla()
    plt.close()

############################################

for width in widths:
    for y in ys:
        df_pr = pd.read_csv(base_path + model_32_name.format(width), delimiter='\t')
        df_sc =pd.read_csv(scra_path + model_32_name.format(width), delimiter='\t')
        plt.plot(range(len(df_pr)),df_pr[y],'-o',label='From pretrained')
        plt.plot(range(len(df_pr)),df_sc[y],'-s',label='From scratch')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel(y)
        plt.title('width multiplier = '+width)
        plt.savefig('/Users/ryan/GitHub/MscProject/analysis/vis_sc_32/'+width.replace('.','_')+'_'+y+'.png',dpi=400)
        plt.cla()
        plt.close()

for width in widths:
    for y in ys:
        df_pr = pd.read_csv(base_path + model_32_name.format(width), delimiter='\t')
        df_sc =pd.read_csv(shor_path + model_16_name.format(width), delimiter='\t')
        plt.plot(range(len(df_pr)),df_pr[y],'-o',label='Sample duration=32')
        plt.plot(range(len(df_pr)),df_sc[y],'-s',label='Sample duration=16')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel(y)
        plt.title('width multiplier = '+width)
        plt.savefig('/Users/ryan/GitHub/MscProject/analysis/vis_16_32/'+width.replace('.','_')+'_'+y+'.png',dpi=400)
        plt.cla()
        plt.close()
