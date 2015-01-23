import redis
import struct
import time
import cPickle

class TelescopeModel(object):
    def __init__(self, host='localhost', db=0):
        self._r = redis.StrictRedis(db=db)
        self._ps = self._r.pubsub(ignore_subscribe_messages=True)
        self._default_channel = 'tm_info'
        self._ps.subscribe(self._default_channel)
         # subscribe to the telescope model info channel

    def _strip(self, str_val):
        if len(str_val) < 8: return None
        ts = struct.unpack('>d',str_val[:8])[0]
        try:
            ret_val = cPickle.loads(str_val[8:])
        except cPickle.UnpicklingError:
            ret_val = str_val[8:]
        return (ret_val,ts)

    def __getattr__(self, key):
        if self._r.exists(key):
            try:
                return self._r.get(key)
                 # assume simple string type
            except redis.ResponseError:
                try:
                    return self._strip(self._r.zrange(key,-1,-1)[0])[0]
                except redis.ResponseError:
                    raise AttributeError
                    
    def __getitem__(self, key):
        if self._r.exists(key):
            try:
                return self._r.get(key)
                 # assume simple string type
            except redis.ResponseError:
                try:
                    return self._strip(self._r.zrange(key,-1,-1)[0])[0]
                except redis.ResponseError:
                    raise AttributeError

    def send_message(self, data, channel=None):
        """Broadcast a message to all telescope model users."""
        if channel is None: channel = self._default_channel
        return self._r.publish(channel, data)

    def get_message(self, channel=None):
        """Get the oldest unread telescope model message."""
        if channel is None: channel = self._default_channel
        msg = self._ps.get_message(channel)
        if msg is not None: msg = msg['data']
        return msg

    def list_keys(self, filter='*', show_counts=False):
        """Return a list of keys currently in the model."""
        key_list = []
        if show_counts:
            keys = self._r.keys(filter)
            for k in keys:
                try:
                    kcount = self._r.zcard(k)
                except redis.ResponseError:
                    kcount = 1
                key_list.append((k,kcount))
        else:
            key_list = self._r.keys(filter)
        return key_list

    def delete(self, key):
        """Remove a key, and all values, from the model."""
        return self._r.delete(key)

    def add(self, key, value, ts=None):
        """Add a new key / value pair to the model."""
        if ts is None: ts = time.time()
        existing_type = self._r.type(key)
        if existing_type != 'none' and existing_type != 'zset':
            self._r.delete(key)
        packed_ts = struct.pack('>d',float(ts))
        return self._r.zadd(key, 0, "{}{}".format(packed_ts,cPickle.dumps(value)))

    def get(self, key, st=None, et=None):
        """Get the value specified by the key from the model."""
        if not self._r.exists(key): return KeyError
        if st is None and et is None:
            return self._strip(self._r.zrange(key,-1,-1)[0])
        elif st == 0 or et == 0:
            return [self._strip(str_val) for str_val in self._r.zrange(key,0,-1)]
        else:
            packed_st = struct.pack('>d',float(st))
            packed_et = struct.pack('>d',float(et))
            ret_vals = self._r.zrangebylex(key,"[{}".format(packed_st),"[{}".format(packed_et))
            return [self._strip(str_val) for str_val in ret_vals]

