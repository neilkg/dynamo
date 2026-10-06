-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0
-- KEYS: record, expiry index, sizes, accounting. All keys share one namespace hash tag.
-- ARGV: operation, logical key, value, expected value, TTL ms, bytes, entries, record bytes.
local operation, key = ARGV[1], ARGV[2]
local limits = 'v1:' .. ARGV[6] .. ':' .. ARGV[7] .. ':' .. ARGV[8]
local configured = redis.call('HGET', KEYS[4], 'limits')
if configured and configured ~= limits then return {-4, ''} end
local clock = redis.call('TIME')
local now = tonumber(clock[1]) * 1000 + math.floor(tonumber(clock[2]) / 1000)
local used = tonumber(redis.call('HGET', KEYS[4], 'bytes') or '0')
-- Bounded by the configured entry limit. Native key expiry removes record contents;
-- this index only retains logical keys and sizes until a frontend sweeps it.
local expired = redis.call('ZRANGEBYSCORE', KEYS[2], '-inf', now)
for _, member in ipairs(expired) do
    used = used - tonumber(redis.call('HGET', KEYS[3], member) or '0')
    redis.call('HDEL', KEYS[3], member)
    redis.call('ZREM', KEYS[2], member)
end
redis.call('HSET', KEYS[4], 'bytes', used, 'limits', limits)
local current = redis.call('GET', KEYS[1])
if operation == 'get' then
    if current then return {1, current} end
    return {0, ''}
end
if operation == 'purge' then return {1, ''} end
if operation == 'cas' and current ~= ARGV[4] then return {0, ''} end
local old_size = tonumber(redis.call('HGET', KEYS[3], key) or '0')
if operation == 'delete' then
    redis.call('DEL', KEYS[1])
    redis.call('HDEL', KEYS[3], key)
    redis.call('ZREM', KEYS[2], key)
    redis.call('HSET', KEYS[4], 'bytes', used - old_size)
    if current then return {1, ''} end
    return {0, ''}
end
local size = string.len(key) + string.len(ARGV[3])
if size > tonumber(ARGV[8]) then return {-2, ''} end
if used - old_size + size > tonumber(ARGV[6]) or
    (old_size == 0 and redis.call('ZCARD', KEYS[2]) >= tonumber(ARGV[7])) then
    return {-1, ''}
end
local ttl = tonumber(ARGV[5])
if ttl <= 0 then
    redis.call('DEL', KEYS[1])
    redis.call('HDEL', KEYS[3], key)
    redis.call('ZREM', KEYS[2], key)
    redis.call('HSET', KEYS[4], 'bytes', used - old_size)
    return {1, ''}
end
redis.call('SET', KEYS[1], ARGV[3], 'PXAT', now + ttl)
redis.call('HSET', KEYS[3], key, size)
redis.call('ZADD', KEYS[2], now + ttl, key)
redis.call('HSET', KEYS[4], 'bytes', used - old_size + size)
-- Bound metadata lifetime even if every frontend disappears. Keep it slightly
-- longer than the last record so it cannot expire before a record's budget entry.
local last = redis.call('ZREVRANGE', KEYS[2], 0, 0, 'WITHSCORES')
for index = 2, 4 do redis.call('PEXPIREAT', KEYS[index], tonumber(last[2]) + 1000) end
return {1, ''}
