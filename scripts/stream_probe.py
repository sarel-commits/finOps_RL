import openai,time,sys,json
sys.path.insert(0,'scripts'); from agent import SAMPLING
c=openai.OpenAI(base_url="http://127.0.0.1:8080/v1",api_key="x",timeout=900)
eb=SAMPLING["nothink"]["extra_body"]; out=[]
P="Write a detailed 400-word essay on cloud cost optimization."
for run,pp in [(1,1.5),(2,0.0)]:
    t0=time.time(); n=0; marks=[]; txt=""
    s=c.chat.completions.create(model="default_model",messages=[{"role":"user","content":P}],max_tokens=400,temperature=0.7,top_p=0.8,extra_body={**eb,"presence_penalty":pp},seed=1,stream=True)
    for ch in s:
        if ch.choices and ch.choices[0].delta.content:
            n+=1; txt+=ch.choices[0].delta.content
            if n%50==0: marks.append((n,round(time.time()-t0,1))); print(run,pp,n,round(time.time()-t0,1),flush=True)
        if time.time()-t0>240: print("ABORT slow",run,n); break
    out.append(dict(run=run,pp=pp,chunks=n,marks=marks,tail=txt[-300:]))
json.dump(out,open(sys.argv[1],'w'),indent=1)
