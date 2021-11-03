class JuliaCompat:
    def __init__(self, src):
        if isinstance(src, str):
            self.parts = self.parse_parts(src)
        else:
            self.parts = list(src)
    def parse_parts(self, src):
        return [self.parse_part(part) for part in src.split(',')]
    def parse_part(self, src):
        src = src.strip()
        if src.startswith('='):
            return self.parse_eq(src[1:])
        elif src.startswith('^'):
            return self.parse_caret(src[1:])
        elif src.startswith('~'):
            return self.parse_tilde(src[1:])
        else:
            return self.parse_caret(src)
    def parse_eq(self, src):
        return Eq(Version(src))
    def parse_tilde(self, src):
        v = Version(src)
        if len(v.parts) == 1:
            return Range(v, v.next_major())
        else:
            return Range(v, v.next_minor())
    def parse_caret(self, src):
        v = Version(src)
        if v.major or v.minor is None:
            return Range(v, v.next_major())
        elif v.minor or v.patch is None:
            return Range(v, v.next_minor())
        else:
            return Range(v, v.next_patch())
    def jlstr(self):
        return ', '.join(p.jlstr() for p in self.parts)
    def isempty(self):
        return all(p.isempty() for p in self.parts)
    def __and__(self, other):
        parts = []
        for p1 in self.parts:
            for p2 in other.parts:
                p3 = p1 & p2
                if not p3.isempty():
                    parts.append(p3)
        return JuliaCompat(parts)
    def __repr__(self):
        return 'JuliaCompat({!r})'.format(self.parts)
    def __contains__(self, v):
        return any(v in p for p in self.parts)

class Version:
    def __init__(self, src):
        if isinstance(src, Version):
            src = src.parts
        elif isinstance(src, str):
            src = src.strip().split('.')
        if 1 <= len(src) <= 3:
            self.parts = tuple(map(int, src))
        else:
            raise Exception("invalid version")
    def __repr__(self):
        return 'Version({!r})'.format(self.parts)
    @property
    def major(self):
        return self.parts[0]
    @property
    def minor(self):
        return self.parts[1] if len(self.parts) > 1 else None
    @property
    def patch(self):
        return self.parts[2] if len(self.parts) > 2 else None
    def jlstr(self):
        return '.'.join(map(str, self.parts))
    def pad_zeros(self):
        return type(self)((self.major or 0, self.minor or 0, self.patch or 0))
    def next_major(self):
        return type(self)((self.major+1, 0, 0))
    def next_minor(self):
        return type(self)((self.major, (self.minor or 0)+1, 0))
    def next_patch(self):
        return type(self)((self.major, self.minor or 0, (self.patch or 0)+1))
    def __eq__(self, other):
        return self.parts == other.parts
    def __le__(self, other):
        return self.parts <= other.parts
    def __lt__(self, other):
        return self.parts < other.parts

class Eq:
    def __init__(self, v):
        self.v = v.pad_zeros()
    def jlstr(self):
        return '=' + self.v.jlstr()
    def isempty(self):
        return False
    def __and__(self, other):
        if self in other:
            return self
        else:
            return Range(Version('0'), Version('0'))
    def __rand__(self, other):
        return self.__and__(other)
    def __contains__(self, v):
        return self.v == v
    def __repr__(self):
        return 'Eq({!r})'.format(self.v)

class Range:
    def __init__(self, v0, v1):
        self.v0 = v0.pad_zeros()
        self.v1 = v1.pad_zeros()
    def jlstr(self):
        if self.v1 == self.v0.next_major():
            if self.v0.major:
                return self.v0.jlstr()
            elif self.v0.minor == 0 and self.v0.patch == 0:
                return '0'
        elif self.v1 == self.v0.next_minor():
            if self.v0.major == 0:
                if self.v0.minor:
                    return self.v0.jlstr()
                elif self.v0.patch == 0:
                    return '0.0'
            return '~' + self.v0.jlstr()
        elif self.v1 == self.v0.next_patch():
            if self.v0.major == 0 and self.v0.minor == 0:
                return self.v0.jlstr()
        raise ValueError("cannot represent range [{}, {}) as a compat entry".format(self.v0.jlstr(), self.v1.jlstr()))
    def isempty(self):
        return self.v1 <= self.v0
    def __and__(self, other):
        if isinstance(other, Range):
            return Range(max(self.v0, other.v0), min(self.v1, other.v1))
        else:
            return NotImplemented
    def __contains__(self, v):
        return self.v0 <= v and v < self.v1
    def __repr__(self):
        return 'Range({!r}, {!r})'.format(self.v0, self.v1)
