import random
#from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
 
def plot_vectors(vectors, labels, label_type, out_dir, seed, plot_name = '2d_vis'):
    '''
    Visualizing vectors by projecting them on 2D
    '''
    model = TSNE(n_components=2, random_state = seed)
    vectors = model.fit_transform(vectors)
    x, y = vectors[:, 0], vectors[:, 1]
    for i, tname in enumerate(labels):
        plt.scatter(x[i], y[i])#, c = cluster_color[domain[tname]])
        plt.annotate(tname, (x[i], y[i]))
    plt.savefig(out_dir+plot_name+'_'+label_type+'.eps', format = 'eps', dpi = 1000)
    plt.show()
    
# def genRandColor():
#     rgbl=[255,0,0]
#     random.shuffle(rgbl)
#     return tuple(rgbl)
#  
# domain = {} #one domain per patient
# cluster_color = {} #color for each unique domain
 
#             cur.execute("select distinct domain from uza11.TS_DEPT where pid = %s and sid = %s and did = %s order by sid desc", (curPid, sid[0], did[0], ))
#             print(cur.fetchall())
#             cur_domain = cur.fetchall()[0]
#             domain[curPid] = cur_domain
#             if cur_domain not in cluster_color:
#                 cluster_color[cur_domain] = genRandColor()
 
