#!/bin/bash
# Run 3 from this Mac: launch a 1xH100 spot instance, upload rl/ + ITBench-Lite, set up, smoke-test, train.
# Needs a valid AWS SSO session:  aws sso login --profile pointfive-sandbox-ai
#
# Run 3 = run 2 + the improvements from the run-2 analysis:
#   --harness improved          train with the eval's improved harness (prompt v2 + step nudges + forced final answer)
#   --reward-exact-weight 0.7   listing extra resources costs more (1 right + 1 extra: 0.2 instead of 0.33)
#   --val-n 30                  synthetic val set at each eval, for choosing a checkpoint (ITBench stays the test)
#   --eval-samples 2            ITBench-Lite eval with 2 samples per scenario (less noise)
#   audit rule narrowed (no more false "label" flags); fresh LoRA (not continued from run 2)
set -euo pipefail
export PATH=~/aws-cli:$PATH AWS_PROFILE=${AWS_PROFILE:-pointfive-sandbox-ai}
R=us-east-2
D=~/finops-eval/rl/.aws-run
HOURS=${HOURS:-10}                       # training deadline (run 2: ~9.6 h, ~$35)
TYPE=${TYPE:-p5.4xlarge}                 # 1x H100 80GB
aws sts get-caller-identity >/dev/null

KEY=finops-rl
aws ec2 describe-key-pairs --region $R --key-names $KEY >/dev/null 2>&1 || \
  aws ec2 import-key-pair --region $R --key-name $KEY --public-key-material fileb://$D/key.pub >/dev/null
ACCT=$(aws sts get-caller-identity --query Account --output text)
SGF=$D/sg_$ACCT                          # security groups are per account; create one on first use
if [ ! -s $SGF ]; then
  VPC=$(aws ec2 describe-vpcs --region $R --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
  aws ec2 create-security-group --region $R --group-name finops-rl --description "finops RL ssh" --vpc-id $VPC \
    --query GroupId --output text > $SGF 2>/dev/null || \
  aws ec2 describe-security-groups --region $R --filters Name=group-name,Values=finops-rl \
    --query 'SecurityGroups[0].GroupId' --output text > $SGF
fi
SG=$(cat $SGF)
MYIP=$(curl -s https://checkip.amazonaws.com)/32
aws ec2 authorize-security-group-ingress --region $R --group-id $SG --protocol tcp --port 22 --cidr $MYIP 2>/dev/null || true
AMI=$(aws ec2 describe-images --region $R --owners amazon \
  --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" "Name=state,Values=available" \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)
echo "AMI $AMI  type $TYPE  sg $SG"
IID=$(aws ec2 run-instances --region $R --image-id $AMI --instance-type $TYPE --key-name $KEY \
  --security-group-ids $SG --instance-market-options '{"MarketType":"spot"}' \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":300,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=finops-rl-run3}]' \
  --query 'Instances[0].InstanceId' --output text)
echo "$R $IID spot" > $D/instance
aws ec2 wait instance-running --region $R --instance-ids $IID
IP=$(aws ec2 describe-instances --region $R --instance-ids $IID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo $IP > $D/ip
sed -i '' "s/^  HostName .*/  HostName $IP/" $D/ssh_config
: > $D/known_hosts
echo "instance $IID at $IP"
for i in $(seq 1 60); do ssh -F $D/ssh_config calib true 2>/dev/null && break; sleep 10; done

cd ~/finops-eval
rsync -az -e "ssh -F $D/ssh_config" --exclude .venv-vf --exclude results --exclude __pycache__ --exclude .aws-run \
  --exclude .pytest_cache rl/ calib:rl/
rsync -az -e "ssh -F $D/ssh_config" data/itbench-lite/snapshots/finops/v0.1-finops-anomaly/ calib:itbench/
ssh -F $D/ssh_config calib 'bash ~/rl/scripts/aws/setup_train.sh' | tail -8

ARGS="--harness improved --reward-exact-weight 0.7 --val-n 30 --eval-samples 2 --tasks-per-step 6 --group 8 \
  --steps 80 --eval-every 8 --fp8-kv"
echo "== smoke test"
ssh -F $D/ssh_config calib "cd ~/rl && ~/.venv-unsloth/bin/python scripts/train_grpo_unsloth.py $ARGS --smoke > ~/smoke.log 2>&1; tail -5 ~/smoke.log"
DEADLINE=$(( $(date +%s) + HOURS * 3600 ))
ssh -F $D/ssh_config calib "~/run_train.sh ~/rl/results/train.log $ARGS --deadline $DEADLINE"
echo "run 3 started on $IP; deadline $(date -r $DEADLINE)"
