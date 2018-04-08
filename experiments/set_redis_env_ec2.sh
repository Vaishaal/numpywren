echo $SHELL
export REDIS_IP=(curl http://169.254.169.254/latest/meta-data/public-ipv4)
export REDIS_PASS=numpywrenosdi2018
export REDIS_PORT=9001

echo "REDIS_IP: $REDIS_IP, REDIS_PASS: $REDIS_PASS, REDIS_PORT: $REDIS_PORT"
