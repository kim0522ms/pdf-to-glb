"""U3D range decoder adapted from svhss/u3d-mesh-extract.
Upstream revision: 71c5f4428ae3bcff9c2521c4e00dd938a703a648.
See THIRD_PARTY_NOTICE.txt and LICENSE-APACHE-2.0.txt.
"""
import struct

MASK32 = 0xFFFFFFFF
uACStaticFull = 0x00000400
uACMaxRange   = uACStaticFull + 0x00003FFF
uACContextBaseShadingID = 1
IFXPROFILE_NOCOMPRESSION = 0x00000004
IFXPROFILE_UNITSSCALE    = 0x00000008
BT_FileHeader, BT_ModifierChain = 0x00443355, 0xFFFFFF14
BT_CLODDecl, BT_CLODBaseMesh, BT_CLODProgr = 0xFFFFFF31, 0xFFFFFF3B, 0xFFFFFF3C

SWAP8 = [0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15]
READCOUNT = [4,3,2,2,1,1,1,1,0,0,0,0,0,0,0,0]
FASTNOTMASK = [0xFFFFFFFF,0x7FFF7FFF,0x3FFF3FFF,0x1FFF1FFF,0x0FFF0FFF]
HALFMASK,NOTHALFMASK = 0x80008000,0x7FFF7FFF
QUARTERMASK,NOTTHREEQUARTERMASK = 0x40004000,0x3FFF3FFF

# ----------------------------------------------------------- histogram
class Histogram:
    __slots__ = ('elephant','count','total')
    def __init__(self, elephant):
        self.elephant = elephant; self.count = [1]; self.total = 1
    def get_total(self): return self.total
    def get_symbol_freq(self, s):
        return self.count[s] if s < len(self.count) else 0
    def get_cum_freq(self, s):
        return sum(self.count[:s]) if s <= len(self.count) else self.total
    def get_symbol_from_freq(self, cf):
        if cf >= self.total: return 0
        acc = sym = 0
        for i,c in enumerate(self.count):
            if acc <= cf: sym = i
            else: break
            acc += c
        return sym
    def add_symbol(self, s):
        if s > 0xFFFF: return
        if self.total >= self.elephant:
            self.count = [x >> 1 for x in self.count]
            self.count[0] += 1; self.total = sum(self.count)
        if s >= len(self.count):
            self.count.extend([0]*(s + 1 - len(self.count)))
        self.count[s] += 1; self.total += 1

# ----------------------------------------------------------- bitstream / range coder
class BitStream:
    def __init__(self, bdata, no_compression=False, elephant=0x1FFF):
        nwords = (len(bdata) + 3)//4 + 4
        buf = bytearray(bdata) + b'\x00'*(nwords*4 - len(bdata))
        self.words = list(struct.unpack('<%dI' % nwords, bytes(buf)))
        self.pos = self.bitoff = 0
        self.local, self.localnext = self.words[0], self.words[1]
        self.high, self.low, self.code, self.underflow = 0xFFFF, 0, 0, 0
        self.nocomp, self.elephant, self.contexts = no_compression, elephant, {}
    def _inc(self):
        self.pos += 1; self.local = self.localnext
        self.localnext = self.words[self.pos + 1]
    def get_bitcount(self): return (self.pos << 5) + self.bitoff
    def seek_ro(self, position):
        self.pos = position >> 5; self.bitoff = position & 31
        self.local = self.words[self.pos]; self.localnext = self.words[self.pos + 1]
    def read_bit(self):
        v = (self.local >> self.bitoff) & 1; self.bitoff += 1
        if self.bitoff >= 32: self.bitoff -= 32; self._inc()
        return v
    def read15(self):
        v = (self.local >> self.bitoff) & MASK32
        if self.bitoff > 17: v |= (self.localnext << (32 - self.bitoff)) & MASK32
        v = (v + v) & MASK32
        v = SWAP8[(v>>12)&0xf] | (SWAP8[(v>>8)&0xf]<<4) | (SWAP8[(v>>4)&0xf]<<8) | (SWAP8[v&0xf]<<12)
        self.bitoff += 15
        if self.bitoff >= 32: self.bitoff -= 32; self._inc()
        return v
    def read_u8(self):
        if self.high == 0xFFFF and self.low == 0 and self.underflow == 0:
            v = (self.local >> self.bitoff) & MASK32
            if self.bitoff > 24: v |= (self.localnext << (32 - self.bitoff)) & MASK32
            v &= 0xFF; self.bitoff += 8
            if self.bitoff >= 32: self.bitoff -= 32; self._inc()
            return v
        v = self._sym_static(uACStaticFull + 256) - 1
        return (SWAP8[v & 0xf] << 4) | SWAP8[v >> 4]
    def read_u16(self): return self.read_u8() | (self.read_u8() << 8)
    def read_u32(self): return self.read_u16() | (self.read_u16() << 16)
    def read_u64(self): return self.read_u32() | (self.read_u32() << 32)
    def read_f32(self): return struct.unpack('<f', struct.pack('<I', self.read_u32()))[0]
    def read_f64(self): return struct.unpack('<d', struct.pack('<Q', self.read_u64()))[0]
    def read_string(self):
        n = self.read_u16(); return bytes(self.read_u8() for _ in range(n))
    def _renorm(self, uState, bc):
        masked = HALFMASK & uState
        while masked == 0 or masked == HALFMASK:
            uState = (((NOTHALFMASK & uState) << 1) | 1) & MASK32
            masked = HALFMASK & uState; bc += 1
        saved = masked
        if bc > 0: bc += self.underflow; self.underflow = 0
        masked = QUARTERMASK & uState; uf = 0
        while masked == 0x40000000:
            uState &= NOTTHREEQUARTERMASK; uState = (uState + uState) & MASK32
            uState |= 1; masked = QUARTERMASK & uState; uf += 1
        self.underflow += uf; uState |= saved
        self.low = uState >> 16; self.high = uState & 0xFFFF
        self.bitoff += bc
        while self.bitoff >= 32: self.bitoff -= 32; self._inc()
    def _fill_code(self):
        position = self.get_bitcount(); self.code = self.read_bit()
        self.bitoff += self.underflow
        while self.bitoff >= 32: self.bitoff -= 32; self._inc()
        self.code = ((self.code << 15) | self.read15()) & MASK32
        self.seek_ro(position)
    def _sym_static(self, context):
        self._fill_code()
        numsym = context - uACStaticFull; total = numsym
        rng = self.high + 1 - self.low
        codecum = (total * (1 + self.code - self.low) - 1) // rng
        value = codecum + 1
        uLow, uHigh = self.low, self.high
        uHigh = uLow - 1 + rng * (value) // total
        uLow  = uLow + rng * (value - 1) // total
        uState = ((uLow << 16) | uHigh) & MASK32
        bc = READCOUNT[((uLow>>12)^(uHigh>>12))&0xF]
        uState &= FASTNOTMASK[bc]; uState = (uState<<bc)&MASK32; uState |= (1<<bc)-1
        bc2 = READCOUNT[((uState>>12)^(uState>>28))&0xF]
        uState &= FASTNOTMASK[bc2]; uState = (uState<<bc2)&MASK32; bc += bc2; uState |= (1<<bc2)-1
        self._renorm(uState, bc)
        return value
    def _sym_dynamic(self, context):
        self._fill_code()
        h = self.contexts.get(context)
        if h is None: h = Histogram(self.elephant); self.contexts[context] = h
        total = h.get_total(); rng = self.high + 1 - self.low
        codecum = (total * (1 + self.code - self.low) - 1) // rng
        value = h.get_symbol_from_freq(codecum)
        vcum, vfreq = h.get_cum_freq(value), h.get_symbol_freq(value)
        uLow, uHigh = self.low, self.high
        uHigh = uLow - 1 + rng * (vcum + vfreq) // total
        uLow  = uLow + rng * vcum // total
        h.add_symbol(value)
        uState = ((uLow << 16) | uHigh) & MASK32
        bc = READCOUNT[((uLow>>12)^(uHigh>>12))&0xF]
        uState &= FASTNOTMASK[bc]; uState = (uState<<bc)&MASK32; uState |= (1<<bc)-1
        self._renorm(uState, bc)
        return value
    def read_symbol(self, context):
        if context == 0: return self._sym_static(uACStaticFull + 256)
        if context > uACStaticFull: return self._sym_static(context)
        return self._sym_dynamic(context)
    def read_compressed_u32(self, context):
        if self.nocomp: return self.read_u32()
        if context and context < uACMaxRange:
            sym = self.read_symbol(context)
            if sym != 0: return sym - 1
            val = self.read_u32()
            if context <= uACStaticFull:
                h = self.contexts.get(context)
                if h is None: h = Histogram(self.elephant); self.contexts[context] = h
                h.add_symbol(val + 1)
            return val
        return self.read_u32()
