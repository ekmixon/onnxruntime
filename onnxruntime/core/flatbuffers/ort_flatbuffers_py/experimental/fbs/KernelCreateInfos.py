# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class KernelCreateInfos(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsKernelCreateInfos(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = KernelCreateInfos()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def KernelCreateInfosBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x4F\x52\x54\x4D", size_prefixed=size_prefixed)

    # KernelCreateInfos
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # KernelCreateInfos
    def NodeIndices(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # KernelCreateInfos
    def NodeIndicesAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint32Flags, o)
        return 0

    # KernelCreateInfos
    def NodeIndicesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return self._tab.VectorLen(o) if o != 0 else 0

    # KernelCreateInfos
    def NodeIndicesIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

    # KernelCreateInfos
    def KernelDefHashes(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint64Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 8))
        return 0

    # KernelCreateInfos
    def KernelDefHashesAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint64Flags, o)
        return 0

    # KernelCreateInfos
    def KernelDefHashesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return self._tab.VectorLen(o) if o != 0 else 0

    # KernelCreateInfos
    def KernelDefHashesIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

def KernelCreateInfosStart(builder): builder.StartObject(2)
def KernelCreateInfosAddNodeIndices(builder, nodeIndices): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(nodeIndices), 0)
def KernelCreateInfosStartNodeIndicesVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def KernelCreateInfosAddKernelDefHashes(builder, kernelDefHashes): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(kernelDefHashes), 0)
def KernelCreateInfosStartKernelDefHashesVector(builder, numElems): return builder.StartVector(8, numElems, 8)
def KernelCreateInfosEnd(builder): return builder.EndObject()
