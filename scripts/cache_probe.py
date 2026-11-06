import openai,time,sys,json
sys.path.insert(0,'scripts'); from agent import SAMPLING
c=openai.OpenAI(base_url="http://127.0.0.1:8080/v1",api_key="x",timeout=900)
eb=SAMPLING["nothink"]["extra_body"]; res=[]
def go(label,prompt,pp,n=40):
    t=time.time(); r=c.chat.completions.create(model="default_model",messages=[{"role":"user","content":prompt}],max_tokens=n,temperature=0.7,top_p=0.8,extra_body={**eb,"presence_penalty":pp},seed=1)
    dt=time.time()-t; d=dict(label=label,pp=pp,tok=r.usage.completion_tokens,s=round(dt,1),tps=round(r.usage.completion_tokens/dt,2)); res.append(d); print(d,flush=True)
P1="Write a detailed essay on cloud cost optimization."; P2="Explain Kubernetes autoscaling in depth."
go("fresh P1 pp=0",P1,0.0); go("fresh P2 pp=1.5",P2,1.5); go("repeat P1 pp=1.5 (cache hit)",P1,1.5); go("repeat P2 pp=0 (cache hit)",P2,0.0)
json.dump(res,open(sys.argv[1],'w'),indent=1)
