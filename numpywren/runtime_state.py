import redis

REDIS_IP = os.environ.get("REDIS_IP", "127.0.0.1")
REDIS_PASS = os.environ.get("REDIS_PASS", "")
REDIS_PORT = os.environ.get("REDIS_PORT", "9001")


class NodeStatus(Enum):
    NOT_READY = 0
    READY = 1
    RUNNING = 2
    POST_OP = 3
    FINISHED = 4


class EdgeStatus(Enum):
    NOT_READY = 0
    READY = 1


class ProgramStatus(Enum):
    SUCCESS = 0
    RUNNING = 1
    EXCEPTION = 2
    NOT_STARTED = 3


class RuntimeState(object):
    def __init__(self, program_hash, ip=REDIS_IP, passw=REDIS_PASS,
                 port=REDIS_PORT, s3=False, s3_bucket=""):
        if s3:
            #TODO: fall back to S3 here.
            raise Exception("S3 fallback not implemented.")
        self._redis_client = redis.StrictRedis(ip, port=port, db=0, password=passw,
                                               socket_timeout=5)

    def set_program_status(status):
        assert isinstance(status, ProgramStatus)
        self._set(self.hash, status)

    def get_program_status():
        status = self._get(self.hash)
        return ProgramStatus(int(status))

    def set_node_status(node_id, status):
        assert isinstance(status, NodeStatus)
        self._set(self._node_key(node_id), status)

    def get_node_status():
        status = self._get(self._node_key(node_id))
        if status is None:
            res = 0
        return NodeStatus(int(status))

    def upload(key, bucket, data):
        client = boto3.client('s3')
        client.put_object(Bucket=bucket, Key=key, Body=data)

    def incr(key, amount):
        return self._redis_client.incr(key, amount=amount)

    def decr(key, amount):
        return self._redis_client.decr(key, amount=amount)

    def set_edge_done(parent_id, child_id):
        ''' Crucial atomic operation needed to insure DAG correctness
            @param key_to_incr - increment this key
            @param condition_key - only do so if this value is 1
            @param ip - ip of redis server
            @param value - the value to bind key_to_set to
        '''
        key_to_incr = self._parents_ready_key(child_id)
        condition_key = self._edge_key(parent_id, child_id)  
        res = 0
        with self._redis_client.pipeline() as pipe:
            while True:
                try:
                    pipe.watch(condition_key)
                    pipe.watch(key_to_incr)
                    current_value = pipe.get(key_to_incr)
                    if current_value is None:
                        current_value = 0
                    current_value = int(current_value)
                    condition_val = pipe.get(condition_key)
                    if condition_val is None:
                        condition_val = 0
                    condition_val = int(condition_val)
                    res = current_value
                    if condition_val == 0:
                        pipe.multi()
                        pipe.incr(key_to_incr)
                        pipe.set(condition_key, 1)
                        t_results = pipe.execute()
                        res = int(t_results[0])
                        assert(t_results[1])
                    break
                except redis.WatchError as e:
                    continue
        return res

    def _get(key):
        return self._redis_client.get(key)

    def _set(key, status):
        self._redis_client.set(key, status.value)

    def _node_key(self, node_id):
        return "{0}_node:{1}".format(self.hash, node_id)

    def _edge_key(self, parent_id, child_id):
        return "{0}_edge:{1}-{2}".format(self.hash, parent_id, child_id)

    def _parents_ready_key(self, child_id):
        return "{0}_parentsready:{1}".format(self.hash, child_id)

