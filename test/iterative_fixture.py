"""Small mirrored thick-wing fixture for operator/adjoint tests (not accuracy)."""
import numpy as np
import jax.numpy as jnp
from panel.differentiable import PanelTopology


def wing_inputs():
    ys = np.linspace(-1., 1., 5)
    ring = np.array([[0.,0.,0.],[.5,0.,.1],[1.,0.,0.],[.5,0.,-.1]])
    xyz = np.concatenate([ring+[0.,y,0.] for y in ys]).tolist()
    faces=[];te=[]
    for i in range(len(ys)-1):
        for j in range(4):
            a,b,c,d = 4*i+j,4*i+(j+1)%4,4*(i+1)+j,4*(i+1)+(j+1)%4
            faces += [[a,b,d],[a,d,c]]
        te.append([4*i+2,4*(i+1)+2])
    for end in (0,4):
        cap=len(xyz);xyz.append([.5,float(ys[end]),0.])
        for j in range(4):
            f=[cap,4*end+j,4*end+(j+1)%4]
            faces.append(f if end==4 else f[::-1])
    downstream=[]
    for y in ys:
        downstream.append(len(xyz));xyz.append([4.,float(y),0.])
    wake=[]
    for i in range(4):
        wake += [[4*i+2,downstream[i],downstream[i+1]], [4*i+2,downstream[i+1],4*(i+1)+2]]
    xyz=np.array(xyz);faces=np.array(faces);wake=np.array(wake);te=np.array(te)
    # Rebuild left panels as exact mirrors; mirroring only a quadrilateral's
    # vertices but choosing a different triangle diagonal is not the same mesh.
    lookup={tuple(x):i for i,x in enumerate(xyz)}
    mirror=np.array([lookup[tuple(x*[1,-1,1])] for x in xyz])
    half=faces[xyz[faces].mean(1)[:,1]>0]
    faces=np.concatenate((half,mirror[half[:,::-1]]))
    whalf=wake[xyz[wake].mean(1)[:,1]>0]
    wake=np.concatenate((whalf,mirror[whalf[:,::-1]]))
    adjacency=[set() for _ in faces];edge_map={}
    for i,f in enumerate(faces):
        for a,b in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])):
            edge_map.setdefault(tuple(sorted((a,b))),[]).append(i)
    assert all(len(v)==2 for v in edge_map.values())
    cuts={tuple(sorted(e)) for e in te}
    for edge,(a,b) in edge_map.items():
        if edge not in cuts:
            adjacency[a].add(b);adjacency[b].add(a)
    neigh=[]
    for i,one in enumerate(adjacency):
        neigh.append(sorted((one|set().union(*(adjacency[j] for j in one)))-{i}))
    width=max(map(len,neigh));neighbors=np.array([r+[i]*(width-len(r)) for i,r in enumerate(neigh)])
    upper=[];lower=[]
    normals=np.cross(xyz[faces[:,1]]-xyz[faces[:,0]],xyz[faces[:,2]]-xyz[faces[:,0]])
    for e in te:
        pair=edge_map[tuple(sorted(e))]
        upper.append(next(i for i in pair if normals[i,2]>0))
        lower.append(next(i for i in pair if normals[i,2]<0))
    hat=np.zeros((5,3));hat[1:4]=np.eye(3)
    wi=np.searchsorted(ys,xyz[wake].mean(1)[:,1])-1
    topology=PanelTopology(*(jnp.asarray(x) for x in (
        faces,wake,te,np.array(upper),np.array(lower),hat,wi,
        np.repeat(np.arange(4)[:,None],2,axis=1),neighbors,np.full(len(faces),10))))
    return dict(vertices=jnp.array(xyz),station_y=jnp.array(ys),topology=topology,
                freestream=jnp.array([1.,0.,.08]),density=jnp.array(1.225),
                reference_area=jnp.array(2.),reference_chord=jnp.array(1.),
                reference_point=jnp.array([.25,0.,0.]))
