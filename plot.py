import numpy as np
import matplotlib.pyplot as plt
import re

plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

model_name = 'cifar'
suffix = 'update_1'
beta_start = 1
beta_end = 10.5
beta_range = [i / 2 for i in range( int( 2*beta_start ), int( 2*beta_end+1 ) )]
beta_range = [int( i ) if i.is_integer() else i for i in beta_range]
acc = []
time = []
for beta in beta_range:
    file_name = 'experimental_results_new/'+model_name+'_inference_exits_beta_'+str(beta)+'.txt'
    with open( file_name ) as f:
        contents = f.readlines()[-3:]
    contents = contents[0] + contents[1] + contents[2]
    acc.append( int( re.findall( 'images: ([0-9]*)', contents )[0] ) )
    time.append( float( re.findall( 'consumed: ([0-9.]*)', contents )[0] ) )
acc = np.array( acc )
time = np.array( time )
file_name = 'experimental_results_new/'+model_name+'_inference_exits_normal.txt'
with open( file_name ) as f:
    contents = f.readlines()[-3:]
contents = contents[0] + contents[1] + contents[2]
acc_normal = int( re.findall( 'images: ([0-9]*)', contents )[0] )
time_normal = float( re.findall( 'consumed: ([0-9.]*)', contents )[0] )

# label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
# color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
# 线型：-  --   -.  :    ,
# marker：.  ,   o   v    <    *    +    1
# plt.figure(figsize=(7, 5))
plt.grid(linestyle="--")  # 设置背景网格线为虚线
ax = plt.gca()
ax.spines['top'].set_visible(False)  # 去掉上边框
ax.spines['right'].set_visible(False)  # 去掉右边框

# TODO: modify the following commands
plt.plot(acc, time, color="blue", label="early-exits", linewidth=1.5)
plt.plot(acc_normal, time_normal, 'go', label="normal training")

# group_labels = ['Top 0-5%', 'Top 5-10%', 'Top 10-20%', 'Top 20-50%', 'Top 50-70%', ' Top 70-100%']  # x轴刻度的标识
# plt.xticks(x, group_labels, fontsize=15, fontweight='bold')  # 默认字体大小为10
plt.xticks(acc, fontsize=15, fontweight='bold')  # 默认字体大小为10
plt.yticks(fontsize=15, fontweight='bold')
# plt.title("heterogeneous data partition", fontsize=15, fontweight='bold')  # 默认字体大小为12
plt.xlabel("accuracy (%)", fontsize=16, fontweight='bold')
plt.ylabel("time (s)", fontsize=16, fontweight='bold')
# plt.xlim(0.9, 6.1)  # 设置x轴的范围
# plt.ylim(60, 95)

# plt.legend()          #显示各曲线的图例
plt.legend(loc=0, numpoints=1)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize=15, fontweight='bold')  # 设置图例字体的大小和粗细

plt.savefig('./plots/accuracy-vs-time-'+model_name+'.png', format='png')
plt.show()