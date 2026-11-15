"""Synthetic Chapter 3 fixtures retained from the 2026-11-09 audit.

No measurements or performance experiments. The fast-bundle test deliberately
varies duration; its outsourcing cost is above the equivalent internal cost.
"""
import numpy as np
from instance import Instance, Mode, Vehicle, DEFAULT_CAPS, STAFF, INF, make_trip_types, build_modes
from model import Window, WindowModel
from paper_model_check import PaperModel
EPS=1e-6
printed_model=PaperModel

def duration_dp(v):
    n = len(v['tasks'])
    p = [0.]*(n+1)
    for k in range(n-1,-1,-1):
        p[k] = min(md.p+p[md.g+1] for md in v['modes'] if md.f == k)
    return [p[k]-p[k+1] for k in range(n)], p

def max_error(model,x):
    A=model.A
    lo=model.lo;hi=model.hi
    rl=model.rlo;rh=model.rhi
    ax=A@x
    linear=max(float(np.maximum(rl-ax,0).max(initial=0)),
               float(np.maximum(ax-rh,0).max(initial=0)),
               float(np.maximum(lo-x,0).max(initial=0)),
               float(np.maximum(x-hi,0).max(initial=0)))
    integral=float(np.abs(x[model.integ.astype(bool)]-np.rint(x[model.integ.astype(bool)])).max(initial=0))
    return dict(linear=linear,integrality=integral)

def literal_objective(M,x):
    """Evaluate the seven printed terms from decoded choices, not from solver objective."""
    W=M.W; inst=W.inst; vs=W.vehs
    parts=dict(work=0.,remaining=0.,transport=0.,rent=0.,late=0.,pickup_wait=0.,hold=0.)
    comp={i:0. for i,v in enumerate(vs) if v['tasks']}
    for j,(i,mdid,t) in enumerate(zip(M.xv,M.xb,M.xs)):
        i=int(i);md=vs[i]['modes'][int(mdid)]; z=x[M.x0+j]
        parts['work']+=md.cost*z
        if md.g==len(vs[i]['tasks'])-1:comp[i]+=(int(t)+md.p)*z
    for j,(i,k) in enumerate(zip(M.ov,M.ok)):
        i=int(i);k=int(k);v=vs[i];z=x[M.o0+j]
        parts['remaining']+=min(md.cost/(md.g-md.f+1) for md in v['modes'] if md.f<=k<=md.g)*z
        pr,_=duration_dp(v)
        comp[i]+=pr[k]*z
        if k==len(v['tasks'])-1:comp[i]+=W.t1*z
    for j,(k,t) in enumerate(zip(M.wk,M.ws)):
        parts['transport']+=inst.trip_types[int(k)].cost*x[M.w0+j]
    for j,d in enumerate(M.days):parts['rent']+=inst.rent_cost_day*x[M.u0+j]
    parts['late']=inst.lateness_cost_min*float(x[M.L0:M.L0+len(vs)].sum())
    for j,(i,t) in enumerate(zip(M.ev,M.es)):
        parts['pickup_wait']+=inst.pickup_wait_cost_min*5*max(0,int(t)-vs[int(i)]['pickup_req'])*x[M.e0+j]
    for j,i in enumerate(M.ne_v):
        parts['pickup_wait']+=inst.pickup_wait_cost_min*5*max(0,W.t1-vs[int(i)]['pickup_req'])*x[M.ne0+j]
    for i,c in comp.items():
        a=vs[i].get('hold_start',W.t0)
        parts['hold']+=inst.hold_cost_day/288*(c-a)
    return dict(total=sum(parts.values()),parts=parts,completion_slots=comp)

def md(k,p=2,c=40,res='gen',kind='I',g=None,park=False):
    prof=np.ones(p,int)
    return Mode(k,k if g is None else g,kind,p,{res:prof},prof if park else np.zeros(p,int),c,
                f'test_{k}_{g}_{p}',1,np.arange(p))

def vd(i=0, tasks=None,modes=None,loc='IN',entry=None,hold=108,due=118,pickup=None):
    tasks=['insp'] if tasks is None else tasks
    modes=[md(0,p=1,kind='Q')] if modes is None and tasks else modes or []
    return dict(vid=i,tasks=tasks,modes=modes,entry_loc=loc,entry_slot=entry,hold=hold,
                deadline=due,pickup_req=pickup,offset=0,hold_start=max(108,entry or 108))

def window(vs,t0=108,t1=132,park_in=3,park_out=2,rent=2,movement=True,gen=4):
    caps=dict(DEFAULT_CAPS);caps['gen']=gen
    if not movement:
        for r in ['supportcar','sup_tow']:caps[r]=0
    ts=make_trip_types()
    for t in ts:t.cost=5.0
    inst=Instance(288,[],caps,STAFF.copy(),{'IN':park_in,'OUT':park_out},rent,25.,ts,
                  lateness_cost_min=20.,pickup_wait_cost_min=20.,hold_cost_day=5000.)
    for v in vs:v['hold_start']=max(t0,v['entry_slot'] if v['entry_slot'] is not None else t0)
    return Window(inst,t0,t1,vs,{},np.zeros(t1-t0,int))

def fixtures():
    cases=[]
    cases.append(('internal_sequence',window([vd(tasks=['wash','insp'],modes=[md(0,2),md(1,1,kind='Q')])]),False))
    om=md(0,4,60,'sup_wash',kind='O',park=True)
    # External handovers use general staff only at departure and return.
    om.res['gen']=np.array([1,0,0,1]);om.park=np.array([1,0,0,1]);om.staff_slots=np.array([0,3])
    cases.append(('outsourced_return_before_qc',window([vd(tasks=['wash','insp'],modes=[om,md(1,1,kind='Q')])]),False))
    cases.append(('recover_then_work',window([vd(loc='OUT',due=112)]),False))
    cases.append(('two_car_shared_trip',window([vd(0,loc='OUT',due=112),vd(1,loc='OUT',due=112)],gen=3),False))
    cases.append(('parking_move_and_rent',window([vd(0,tasks=[],pickup=None),vd(1,tasks=[],entry=110,pickup=None)],park_in=1,park_out=0,rent=2),False))
    cases.append(('infeasible_two_waiters_one_space',window([vd(0,tasks=[]),vd(1,tasks=[])],park_in=1,park_out=0,rent=0,movement=False),False))
    cases.append(('shared_worker_capacity',window([vd(0,modes=[md(0,3)],due=111),vd(1,modes=[md(0,3)],due=111)],gen=1,movement=False),False))
    cases.append(('known_customer_pickup',window([vd(tasks=[],pickup=114)]),False))
    cases.append(('unknown_customer_pickup',window([vd(tasks=[],pickup=None)]),False))
    cases.append(('approval_pending',window([vd(hold=INF,due=INF)]),False))
    cases.append(('end_of_business_day',window([vd(modes=[md(0,2)],hold=215,due=216)],t0=215,t1=225),False))
    cases.append(('approval_received_112',window([vd(hold=112)]),False))
    vv=vd(tasks=['body_h','paint_h','insp'],modes=build_modes(['body_h','paint_h','insp']),hold=132,due=132)
    cases.append(('current_catalog_bundle_all_deferred',window([vv]),False))
    vv=vd(tasks=['A','B','insp'],modes=[md(0,4,40),md(1,4,40),md(2,1,10,kind='Q'),md(0,6,100,g=1,kind='O')],hold=132,due=132)
    cases.append(('fast_bundle_counterexample',window([vv]),True))
    cases.append(('same_block_exchange',window([vd(0,tasks=[],pickup=108),vd(1,entry=109)],park_in=1,park_out=0,rent=0,movement=False),False))
    return cases

def mutations():
    results=[]
    def check(label,M,P,x,tag=None):
        ax=P.A@x
        r=np.maximum(P.rlo-ax,0)+np.maximum(ax-P.rhi,0)
        tags={str(t):float(r[np.array(P.tags)==t].max(initial=0)) for t in np.unique(P.tags)}
        err=max_error(P,x)
        if tag:assert tags.get(tag,0)>EPS,(label,tags)
        else:assert max(err.values())>EPS,(label,err)
        results.append(dict(label=label,expected_tag=tag,violations={k:v for k,v in tags.items() if v>EPS},residual=err))
    W=window([vd(tasks=['A','insp'],modes=[md(0,2),md(1,1,kind='Q')])])
    M=WindowModel(W);P=printed_model(M)
    good=M.encode(dict(modes=[(0,0,108),(0,1,110)],trips=[],exits=[],rent={}))
    assert max(max_error(P,good).values())<EPS
    x=good.copy();x[M.x0+np.where((M.xb==0)&(M.xs==108))[0][0]]=0;check('missing_work_coverage',M,P,x,'3.2')
    x=good.copy();x[M.oidx[(0,0)]]=1;x[M.oidx[(0,1)]]=0;check('non_suffix_deferral',M,P,x,'3.3')
    x=M.encode(dict(modes=[(0,0,109),(0,1,108)],trips=[],exits=[],rent={}));check('successor_before_predecessor',M,P,x,'3.4')
    x=good.copy();x[M.y0]+=1;check('broken_vehicle_flow',M,P,x,'3.6')
    W=window([vd(tasks=[],pickup=114)]);M=WindowModel(W);P=printed_model(M)
    x=M.encode(dict(modes=[],trips=[],exits=[],rent={}));x[M.ne0]=0;check('missing_delivery_or_carry_choice',M,P,x,'3.5')
    W=window([vd(loc='OUT')]);M=WindowModel(W,m=3);P=printed_model(M)
    x=M.encode(dict(modes=[(0,0,114)],trips=[(0,1,114)],exits=[],rent={}));check('within_block_work_before_arrival',M,P,x,'3.7')
    W=window([vd(loc='OUT',tasks=[])],park_out=0,rent=2);M=WindowModel(W);P=printed_model(M)
    x=M.encode(dict(modes=[],trips=[],exits=[],rent={}))
    x[M.sg0:M.sg0+2*M.NB]=0
    check('parking_capacity_overflow',M,P,x,'3.9')
    x=M.encode(dict(modes=[],trips=[],exits=[],rent={0:3}));check('rent_above_upper_bound',M,P,x)
    W=window([vd()]);M=WindowModel(W);P=printed_model(M)
    x=M.encode(dict(modes=[],trips=[(0,1,114)],exits=[],rent={}));check('retrieve_car_not_in_external_parking',M,P,x,'3.11')
    W=window([vd(loc='OUT')]);M=WindowModel(W);P=printed_model(M)
    x=M.encode(dict(modes=[],trips=[(0,1,114)],exits=[],rent={}));j=np.where((M.wk==1)&(M.ws==114))[0][0];x[M.w0+j]=0;check('trip_count_participation_mismatch',M,P,x,'3.12')
    W=window([vd(0,modes=[md(0,3)]),vd(1,modes=[md(0,3)])],gen=1);M=WindowModel(W);P=printed_model(M)
    x=M.encode(dict(modes=[(0,0,108),(1,0,108)],trips=[],exits=[],rent={}));check('shared_worker_double_booking',M,P,x,'3.13')
    W=window([vd(hold=132,due=132)]);M=WindowModel(W);P=printed_model(M)
    x=M.encode(dict(modes=[],trips=[],exits=[],rent={}));x[M.L0]=0;check('unreported_lateness',M,P,x,'3.14')
    W=window([vd()]);M=WindowModel(W);P=printed_model(M)
    x=M.encode(dict(modes=[],trips=[],exits=[],rent={}));x[M.oidx[(0,0)]]=0.5;check('fractional_binary_variable',M,P,x)
    return results
