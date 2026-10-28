import os, shutil, subprocess, sys
base=r'D:\3D\Projects\panel\runs\ref_audit'
for d in ['wing_only_276','elliptic_1028','elliptic_1610','wing_only_v2']:
    p=os.path.join(base,d)
    if os.path.isdir(p): shutil.rmtree(p, ignore_errors=True)
cases={'wing_only_276':r'D:\3D\Projects\panel\runs\half_cad_wing_only_v2',
       'elliptic_1028':r'D:\3D\Projects\panel\runs\geom_robust\mesh\elliptic_s100',
       'elliptic_1610':r'D:\3D\Projects\panel\runs\geom_robust\mesh\elliptic_s075'}
env=dict(os.environ); env['TMP']=env['TEMP']=env['TMPDIR']=r'D:\3D\Projects\panel\runs\pipwork'
for tag, case in cases.items():
    out=os.path.join(base,tag)
    print('RUN', tag, flush=True)
    p=subprocess.run([sys.executable,'-m','panel.audit_reference','--case',case,'--out',out],
                     cwd=r'D:\3D\Projects\panel\runs\pr49_git', env=env,
                     capture_output=True, text=True)
    print('  rc', p.returncode, (p.stdout or '')[-300:], (p.stderr or '')[-300:], flush=True)