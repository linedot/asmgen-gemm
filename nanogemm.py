from asmgen.asmblocks.noarch import asmgen, greg, freg, vreg
from asmgen.asmblocks.noarch import asm_data_type
from asmgen.asmblocks.noarch import reg_tracker

from asmgen.cppgen.types import c_data_types

from enum import Enum,unique
from typing import Type

@unique
class mem_use_type(Enum):
    SAMEDATA=1
    L1=2
    CONTIGUOUS=3

@unique
class bvec_strategy_type(Enum):
    DIST1_BOFF = 1
    DIST1_INC  = 2
    FMAIDX     = 3
    FMAVF      = 4
    NOLOAD     = 5

@unique
class avec_strategy_type(Enum):
    POSTLOAD = 1
    PRELOAD = 2

class kernel_layout:
    bvec_strat = bvec_strategy_type.DIST1_BOFF
    avec_strat = avec_strategy_type.POSTLOAD

class prefetch_options:
    a_init_count = 16
    b_init_count = 10
    c_init_count = 8
    cl_size = 64

class gemm_tracker:
    def __init__(self, rt : reg_tracker, 
                 avec_strat : avec_strategy_type, 
                 vec_in_mr : int, nr : int,
                 bfreg_count : int = -1,
                 bvreg_count : int = -1,
                 aareg : int = 0,
                 bareg : int = 1,
                 careg : int = 2):
        # empty set
        self.avreg_first = 0
        self.avreg_count = 0
        self.bvreg_first = 0
        self.bvreg_count = 0
        self.bfreg_first = 0
        self.bfreg_count = bfreg_count
        self.cvreg_first = 0
        self.cvreg_count = 0
    
        self.vlreg = 0

        self.rt = rt
        self.aareg = aareg
        self.bareg = bareg
        self.careg = careg
        self.rt.reserve_specific_greg(aareg)
        self.rt.reserve_specific_greg(bareg)
        self.rt.reserve_specific_greg(careg)

        self.cvreg_count = vec_in_mr*nr
        self.cvreg_first = self.rt.max_vregs - self.cvreg_count
        self.avreg_first = 0
        if avec_strategy_type.POSTLOAD == avec_strat:
            self.avreg_count = vec_in_mr
        elif avec_strategy_type.PRELOAD == avec_strat:
            self.avreg_count = vec_in_mr*2
        self.bvreg_first = self.avreg_count+self.avreg_first
        self.bvreg_count = self.cvreg_first-self.bvreg_first
        if bvreg_count > 0:
            self.bvreg_count = min(self.bvreg_count, bvreg_count)

        vreg_range = [v for v in range(self.avreg_first, self.avreg_first + self.avreg_count)] +\
                     [v for v in range(self.bvreg_first, self.bvreg_first + self.bvreg_count)]

        assert all(v < self.rt.max_vregs for v in vreg_range), f"Some vector registers chosen over max vec. register ({self.rt.max_vregs})"

        for vreg in vreg_range:
            self.rt.reserve_specific_vreg(vreg)

        # other state
        self.oldareg = self.aareg
        self.oldbreg = self.bareg
        self.cur_avreg = 0
        self.avreg_offset = 0
        self.aoffset = 0
        self.boffset = 0
        self.cur_bvreg = 0
        self.cur_bfreg = 0


    def avreg(self,i):
        assert i < self.avreg_count, f"Requested A vector register nr. {i}. Only {self.avreg_count} available"
        return self.avreg_first+i

    def avreg_rot(self,i):
        idx = i % self.avreg_count
        return self.avreg(idx)

    def bvreg(self,i):
        assert i < self.bvreg_count, f"Requested B vector register nr. {i}. Only {self.bvreg_count} available"
        return self.bvreg_first+i

    def bfreg(self,i):
        assert i < self.bfreg_count, f"Requested B scalar register nr. {i}. Only {self.bfreg_count} available"
        return self.bfreg_first+i

    def bvreg_rot(self,i):
        idx = i % self.bvreg_count
        return self.bvreg(idx)

    def bfreg_rot(self,i):
        idx = i % self.bfreg_count
        return self.bfreg(idx)

    def cvreg(self,i):
        assert i < self.cvreg_count, f"Requested C vector register nr. {i}. Only {self.cvreg_count} available"
        return self.cvreg_first+i

    def get_avreg_count(self):
        return self.avreg_count

    def get_bvreg_count(self):
        return self.bvreg_count

    def get_bfreg_count(self):
        return self.bfreg_count

    def get_cvreg_count(self):
        return self.cvreg_count


# ========================== #
# address register advancing #
# ========================== #

def advance_vecaddr_voffset(offset, mem_use) -> int:
    if mem_use_type.SAMEDATA == mem_use:
        # Same address always
        return offset
    if mem_use_type.L1 == mem_use or \
       mem_use_type.CONTIGUOUS == mem_use:
        return offset+1

    raise RuntimeError(f"Invalid mem_use : {mem_use}")

def advance_b_offset(offset, asm, strat, mem_use, dt) -> int:
    assert isinstance(strat, bvec_strategy_type), f"strat not instance of bvec_strategy_type. strat: {strat}"
    if bvec_strategy_type.DIST1_INC == strat or bvec_strategy_type.NOLOAD == strat:
        return offset
    elif bvec_strategy_type.DIST1_BOFF == strat or bvec_strategy_type.FMAVF == strat:
        # For now advance by a double
        if mem_use == mem_use_type.SAMEDATA:
            return offset
        elif mem_use == mem_use_type.L1:
            return offset+dt.value
        elif mem_use == mem_use_type.CONTIGUOUS:
            return offset+dt.value
        raise RuntimeError(f"Invalid mem_use : {mem_use}")
    elif bvec_strategy_type.FMAIDX == strat:
        if mem_use == mem_use_type.SAMEDATA:
            return offset
        elif mem_use == mem_use_type.L1:
            return offset+1
        elif mem_use == mem_use_type.CONTIGUOUS:
            return offset+1
        raise RuntimeError(f"Invalid mem_use : {mem_use}")
    raise RuntimeError(f"Invalid bvec_strat : {strat}")


class gemm_params:
    def __init__(self):
        self.quirks = 0

# ==================== #
# adding vector offset #
# ==================== #

def add_voff(asm : asmgen, areg : greg, aoffset, vlreg, datatype):
    asmblock = ""
    # TODO: actually check if add_greg_voff is available instead of hardcoding rvv
    if asm.has_add_greg_voff:
        asmblock += asm.add_greg_voff(areg, aoffset, datatype)
    else:
        asmblock += asm.add_greg_greg(areg, areg, vlreg)
    return asmblock

# ============================== #
# loading 1 vector from A matrix #
# ============================== #

def load_a_vec(asm : asmgen, grt : gemm_tracker,
               mem_use : mem_use_type, datatype : asm_data_type):
    asmblock  = asm.load_vector_voff(asm.greg(grt.aareg), 
                                     grt.aoffset, 
                                     asm.vreg(grt.cur_avreg),
                                     datatype)
    grt.aoffset = advance_vecaddr_voffset(grt.aoffset, mem_use)
    # There is a limit to immediate offsets, i.e for SVE you can offset -8 to 7 vectors
    # if we roll over this limit, add the offset to the a address register and 
    if grt.aoffset > asm.max_load_voff:
        asmblock += add_voff(asm, asm.greg(grt.aareg), grt.aoffset,
                             asm.greg(grt.vlreg), datatype)
        grt.aoffset = 0
    return asmblock

# ============================= #
# loading 1 value from B matrix #
# ============================= #

def load_b(asm : asmgen, grt : gemm_tracker,
           layout : kernel_layout,
           mem_use : mem_use_type, datatype : asm_data_type):
    asmblock = ""
    boffcompare = 0
    boffadd = lambda reg,off : ""

    elements_in_vector = asm.simd_size//datatype.value


    if bvec_strategy_type.DIST1_INC == layout.bvec_strat:
        asmblock += asm.load_vector_dist1_inc(asm.greg(grt.bareg), 
                                              grt.boffset, 
                                              asm.vreg(grt.bvreg_rot(grt.cur_bvreg)),
                                              datatype)
        # Not important as offset stays
        boffcompare = asm.max_load_immoff(datatype)
        boffadd = lambda greg, boffset : asm.add_greg_imm(greg, boffset)
    elif bvec_strategy_type.DIST1_BOFF == layout.bvec_strat:
        asmblock += asm.load_vector_dist1_boff(asm.greg(grt.bareg),
                                               grt.boffset, 
                                               asm.vreg(grt.bvreg_rot(grt.cur_bvreg)),
                                               datatype)
        boffcompare = asm.max_load_immoff(datatype)
        boffadd = lambda greg, boffset : asm.add_greg_imm(greg, boffset)
    elif bvec_strategy_type.FMAIDX == layout.bvec_strat:
        asmblock += asm.load_vector_voff(asm.greg(grt.bareg),
                                         grt.boffset, 
                                         asm.vreg(grt.bvreg_rot(grt.cur_bvreg//elements_in_vector)),
                                         datatype)
        boffcompare = asm.max_load_voff
        boffadd = lambda greg, voffset : asm.add_greg_voff(greg, voffset, datatype)
    elif bvec_strategy_type.FMAVF == layout.bvec_strat:
        asmblock += asm.load_scalar_immoff(asm.greg(grt.bareg),
                                           grt.boffset, 
                                           asm.freg(grt.bfreg_rot(grt.cur_bfreg)),
                                           datatype)
        boffcompare = asm.max_fload_immoff(datatype)
        boffadd = lambda greg, boffset : asm.add_greg_imm(greg, boffset)

    if bvec_strategy_type.NOLOAD != layout.bvec_strat:
        grt.boffset = advance_b_offset(grt.boffset, asm, layout.bvec_strat, mem_use, datatype)
        if grt.boffset > boffcompare:
            # No need to worry about vector offset on RVV, because we're not supporting FMAIDX
            asmblock += boffadd(asm.greg(grt.bareg), grt.boffset)
            grt.boffset = 0
    return asmblock

# =========================#
# Updating the C microtile #
# =========================#

def update_c_tile(asm : asmgen, rt : reg_tracker, grt : gemm_tracker,
                  layout, mem_use,
                  free_vregs,
                  alphareg,
                  betareg,
                  vectors_in_mr, nr,
                  beta0,
                  datatype):
    c_tile_store_queue : list[int] = []
    coffset=0
    csoffset=0
    # register holding address for storing C tile
    casreg = rt.reserve_any_greg()
    # copy from c source reg
    asmblock = asm.mov_greg(asm.greg(grt.careg), asm.greg(casreg))
    # TODO: Deduplicate code for c store in and after loop
    # TODO: Deduplicate code for advancing offsets (also in the code above)
    for i in range(nr):
        for j in range(vectors_in_mr):
            tile_idx = grt.cvreg(i*vectors_in_mr+j)

            # Start storing if we run out of free regs or already halfway through
            if 0 == len(free_vregs) or len(c_tile_store_queue) > ((nr*vectors_in_mr)//2):
                csvreg = c_tile_store_queue.pop(0)
                asmblock += asm.store_vector_voff(asm.greg(casreg),
                                      csoffset,
                                      asm.vreg(csvreg),
                                      datatype)
                free_vregs.append(csvreg)
                csoffset = advance_vecaddr_voffset(csoffset, mem_use)
                if csoffset > asm.max_load_voff:
                    asmblock += add_voff(asm, asm.greg(casreg), csoffset, 
                                         asm.greg(grt.vlreg), datatype)
                    csoffset = 0


            cvreg = 0
            # get a free vreg from the beginning of the list
            if not beta0:
                cvreg = free_vregs.pop(0)
                asmblock += asm.load_vector_voff(asm.greg(grt.careg), 
                                                 coffset, 
                                                 asm.vreg(cvreg),
                                                 datatype)
                coffset = advance_vecaddr_voffset(coffset, mem_use)
                if coffset > asm.max_load_voff:
                    asmblock += add_voff(asm, asm.greg(grt.careg), coffset, 
                                         asm.greg(grt.vlreg), datatype)
                    coffset = 0

            if layout.bvec_strat == bvec_strategy_type.FMAVF:
                if not beta0:
                    asmblock += asm.fmul_vf(asm.vreg(cvreg),
                                            asm.freg(betareg),
                                            asm.vreg(cvreg),
                                            datatype)
                    asmblock += asm.fma_vf(asm.vreg(tile_idx),
                                           asm.freg(alphareg),
                                           asm.vreg(cvreg),
                                           datatype)
                    c_tile_store_queue.append(cvreg)
                    free_vregs.append(tile_idx)
                else:
                    asmblock += asm.fmul_vf(asm.vreg(tile_idx),
                                            asm.freg(alphareg),
                                            asm.vreg(tile_idx),
                                            datatype)
                    c_tile_store_queue.append(tile_idx)
            else:
                if not beta0:
                    asmblock += asm.fmul(asm.vreg(cvreg),
                                         asm.vreg(betareg),
                                         asm.vreg(cvreg),
                                         datatype)
                    asmblock += asm.fma(asm.vreg(alphareg),
                                        asm.vreg(tile_idx),
                                        asm.vreg(cvreg),
                                         datatype)
                    c_tile_store_queue.append(cvreg)
                    free_vregs.append(tile_idx)
                else:
                    asmblock += asm.fmul(asm.vreg(tile_idx),
                                         asm.vreg(alphareg),
                                         asm.vreg(tile_idx),
                                         datatype)
                    c_tile_store_queue.append(tile_idx)
            # TODO: can tile_idx == cvreg happen? (would already cause trouble before, 
            #       but check anyways)
            
            # Alternatively:
            # asmblock += asm.fmul(asm.vreg(tile_idx),
            #                      asm.vreg(alphareg),
            #                      asm.vreg(tile_idx),
            #                      datatype)
            # asmblock += asm.fma(asm.vreg(betareg),
            #                     asm.vreg(cvreg),
            #                     asm.vreg(tile_idx),
            #                      datatype)
            # c_tile_store_queue.append(tile_idx)
            # free_vregs.append(cvreg)

    for csvreg in c_tile_store_queue:
        asmblock += asm.store_vector_voff(asm.greg(casreg),
                              csoffset,
                              asm.vreg(csvreg),
                              datatype)
        csoffset = advance_vecaddr_voffset(csoffset, mem_use)
        if csoffset > asm.max_load_voff:
            asmblock += add_voff(asm, asm.greg(casreg), csoffset, 
                                 asm.greg(grt.vlreg), datatype)
            csoffset = 0
    rt.unuse_greg(casreg)

    return asmblock


# =================================== #
# Inner microkernel (one k-iteration) #
# =================================== #


def inner_kernel(asm : asmgen, grt : gemm_tracker, 
                 layout : kernel_layout,
                 mem_use : mem_use_type,
                 vectors_in_mr : int, 
                 nr : int,
                 no_unroll : bool,
                 unroll_factor : int,
                 wrapup_a : bool,
                 wrapup_b : bool,
                 datatype : asm_data_type):
    asmblock = ""
    ignore_a_load = wrapup_a
    elements_in_vector = asm.simd_size//datatype.value
    # for each b value we simd-multiply+add mr values
    for i in range(nr):
        for j in range(vectors_in_mr):
            tile_idx = grt.cvreg(i*vectors_in_mr+j)


            if avec_strategy_type.PRELOAD == layout.avec_strat and \
                    not no_unroll and \
                    not ignore_a_load:
                if ignore_a_load:
                    print("WTF")
                # On first b vector load the next a vectors
                if i == 0:
                    grt.cur_avreg = (grt.avreg_offset+j+vectors_in_mr)%grt.get_avreg_count()
                    asmblock += load_a_vec(asm, grt, 
                                           mem_use, datatype)
            # FMA HERE
            avreg_id = (grt.avreg_offset+j)
            if layout.bvec_strat in [bvec_strategy_type.DIST1_BOFF,bvec_strategy_type.DIST1_INC,bvec_strategy_type.NOLOAD]:
                asmblock += asm.fma(asm.vreg(grt.avreg(avreg_id)),
                                    asm.vreg(grt.bvreg_rot(grt.cur_bvreg)), 
                                    asm.vreg(tile_idx), datatype)
            elif bvec_strategy_type.FMAIDX == layout.bvec_strat:
                asmblock += asm.fma_idx(asm.vreg(grt.avreg(avreg_id)),
                                        asm.vreg(grt.bvreg_rot(grt.cur_bvreg//elements_in_vector)),
                                        asm.vreg(tile_idx), grt.cur_bvreg % elements_in_vector, datatype)
            elif bvec_strategy_type.FMAVF == layout.bvec_strat:
                asmblock += asm.fma_vf(asm.vreg(grt.avreg(avreg_id)),
                                       asm.freg(grt.bfreg_rot(grt.cur_bfreg)),
                                       asm.vreg(tile_idx), datatype)

        if bvec_strategy_type.FMAVF == layout.bvec_strat:
            ignore_b_load = wrapup_b and (grt.cur_bfreg > unroll_factor*nr-grt.get_bfreg_count()-1)
        else:
            ignore_b_load = wrapup_b and (grt.cur_bvreg > unroll_factor*nr-grt.get_bvreg_count()-1)

        if not ignore_b_load:
            asmblock += load_b(asm=asm, grt=grt, layout=layout,
                               mem_use=mem_use, datatype=datatype)
        else:
            asmblock += f"// bfreg={grt.cur_bfreg}, unroll_factor*nr-grt.get_bfreg_count()-1={unroll_factor*nr-grt.get_bfreg_count()-1}\n"
        grt.cur_bvreg += 1
        grt.cur_bfreg += 1

    if not ignore_a_load:
        if avec_strategy_type.PRELOAD == layout.avec_strat and not no_unroll:
            # Cycling through available a registers. Right now for a MvxN kernel we use 2M vectors for A and
            # alternate through the first M and the second M registers
            grt.avreg_offset = (grt.avreg_offset+vectors_in_mr)%grt.get_avreg_count()
        elif (avec_strategy_type.POSTLOAD == layout.avec_strat) or no_unroll:
            for j in range(vectors_in_mr):
                grt.cur_avreg = (grt.avreg_offset+j)
                asmblock += load_a_vec(asm, grt, mem_use, datatype)

    return asmblock

# ================================= #
# clean up after k-loop             #
# (finalize matrix-matrix multiply) #
# ================================= #

def finalize_mm(asm : asmgen, grt :gemm_tracker, 
                layout : kernel_layout, unroll_factor : int,
                nr : int,
                datatype : asm_data_type) -> str:
    asmblock = ""
    bvreg_rem = 0
    reg_count = grt.get_bvreg_count()
    elements_in_vector = asm.simd_size//datatype.value
    if layout.bvec_strat in [bvec_strategy_type.DIST1_BOFF, bvec_strategy_type.DIST1_INC, bvec_strategy_type.NOLOAD]:
        # Here we are using as many vregs as we have reserved
        bvreg_rem = grt.cur_bvreg % grt.get_bvreg_count()
        # If we have more registers available than the loop needs, this is not a problem
        if grt.cur_bfreg > grt.get_bfreg_count():
            bvreg_rem = 0
    elif bvec_strategy_type.FMAIDX == layout.bvec_strat:
        # Here cur_bvreg actually tracks number of b values, so the number of 
        bvreg_rem = (grt.cur_bvreg // elements_in_vector) % grt.get_bvreg_count() + grt.cur_bvreg % elements_in_vector
    elif bvec_strategy_type.FMAVF == layout.bvec_strat:
        bvreg_rem = grt.cur_bfreg % grt.get_bfreg_count()
        # If we have more registers available than the loop needs, this is not a problem
        if grt.cur_bfreg > grt.get_bfreg_count():
            bvreg_rem = 0
        # If we only unroll once it's also not a problem
        if nr == grt.cur_bfreg and 1 == unroll_factor:
            bvreg_rem = 0
        reg_count = grt.get_bfreg_count()
    if 0 != bvreg_rem:
        print(f"WARNING: {reg_count} B vector registers not rolling over cleanly with {unroll_factor} unrolls, remainder:{bvreg_rem}")
    # End of unrolled kernel. If the offsets didn't cleanly roll over, add them to the address registers
    if grt.aoffset > 0:
        asmblock += add_voff(asm, asm.greg(grt.aareg), grt.aoffset, 
                             asm.greg(grt.vlreg), datatype)
        grt.aoffset = 0
    if grt.boffset > 0:
        asmblock += asm.add_greg_imm(asm.greg(grt.bareg), grt.boffset)
        grt.boffset = 0

    # Reset current a/b vreg indices
    grt.cur_avreg = 0
    grt.cur_bfreg = 0
    grt.cur_bvreg = 0
    return asmblock


# ================================#
# Load addresses, preload vectors #
# and values, prefetch memory     #
# ================================#

def memoryinit(asm : asmgen, grt : gemm_tracker, layout : kernel_layout,
               nr : int,
               mem_use : mem_use_type,
               prefetch : prefetch_options,
               preload_a : int, preload_b : int,
               datatype : asm_data_type):
    assert isinstance(layout, kernel_layout), f"Not a kernel layout: {layout}"
    assert isinstance(datatype, asm_data_type), f"Not an asm_data_type: {datatype}"

    rt = grt.rt
    asmblock  = asm.load_pointer(asm.greg(grt.aareg), "a")
    asmblock += asm.load_pointer(asm.greg(grt.bareg), "b")
    asmblock += asm.load_pointer(asm.greg(grt.careg), "c")

    if mem_use_type.L1 == mem_use:
        oldareg_idx = rt.reserve_any_greg()
        oldbreg_idx = rt.reserve_any_greg()
        oldareg = asm.greg(oldareg_idx)
        oldbreg = asm.greg(oldbreg_idx)
        grt.oldareg = oldareg_idx
        grt.oldbreg = oldbreg_idx
        asmblock += asm.mov_greg(asm.greg(grt.aareg), oldareg)
        asmblock += asm.mov_greg(asm.greg(grt.bareg), oldbreg)

    # ================================= #
    #             ISA QUIRKS            #
    # ================================= #

    # SVE: set p0 to true
    if asm.__class__.__name__ == "sve":
        asmblock += asm.ptrue(asm.preg(0),datatype)

    # RVV: vsetvlmax for the datatype
    if asm.__class__.__name__.startswith("rvv"):
        sparereg_idx = rt.reserve_any_greg()
        sparereg = asm.greg(sparereg_idx)
        asmblock += asm.vsetvlmax(sparereg, datatype)
        rt.unuse_greg(sparereg_idx)

    if asm.__class__.__name__.startswith("rvv"):
        vlreg_idx = rt.reserve_any_greg()
        vlreg = asm.greg(vlreg_idx)
        asmblock += asm.simd_size_to_greg(vlreg, datatype)
        grt.vlreg = vlreg_idx

    # ================================= #
    #             PREFETCHING           #
    # ================================= #

    list_bac =[(grt.aareg, prefetch.a_init_count),
              (grt.bareg, prefetch.b_init_count),
              (grt.careg, prefetch.c_init_count)]
    for (areg,count) in list_bac:
        offset = 0
        restore_areg = False
        tmpreg = rt.reserve_any_greg()
        for _ in range(count):
            asmblock += asm.prefetch_l1_boff(asm.greg(areg), offset)
            offset += prefetch.cl_size
            if offset > asm.max_prefetch_offset:
                if not restore_areg:
                    asmblock += asm.mov_greg(asm.greg(areg), asm.greg(tmpreg))
                asmblock += asm.add_greg_imm(asm.greg(areg), offset)
                restore_areg = True
        if restore_areg:
            asmblock += asm.mov_greg(asm.greg(tmpreg), asm.greg(areg))
        rt.unuse_greg(tmpreg)

    # ================================= #
    #             PRELOADING            #
    # ================================= #


    # preloading a
    offset = 0
    for i in range(preload_a):
        asmblock += asm.load_vector_voff(
                asm.greg(grt.aareg),
                offset,
                asm.vreg(grt.avreg(i)),
                datatype)
        offset += 1
        if offset > asm.max_load_voff:
            asmblock += add_voff(asm, asm.greg(grt.aareg), offset, 
                                 asm.greg(grt.vlreg), datatype)
            offset = 0

    if 0 != offset:
        asmblock += add_voff(asm, asm.greg(grt.aareg), offset, 
                             asm.greg(grt.vlreg), datatype)
        offset = 0

    # preloading b
    short_b_offset = 0
    elements_in_vector = asm.simd_size//datatype.value
    tmpreg = rt.reserve_any_greg()
    if bvec_strategy_type.FMAIDX == layout.bvec_strat:
        preload_b = preload_b//elements_in_vector
    for i in range(preload_b):
        asmblock += load_b(asm=asm, grt=grt, layout=layout,
                           mem_use=mem_use, datatype=datatype)

        if bvec_strategy_type.FMAIDX == layout.bvec_strat:
            grt.cur_bfreg += elements_in_vector
            grt.cur_bvreg += elements_in_vector
        else:
            grt.cur_bfreg += 1
            grt.cur_bvreg += 1

        if nr-1 == i and preload_b > nr and grt.boffset > 0:
            asmblock += asm.mov_param_to_greg("iterations",asm.greg(tmpreg))
            asmblock += asm.jzero(asm.greg(tmpreg), "short_b_fixup")
            short_b_offset = grt.boffset

    if bvec_strategy_type.NOLOAD != layout.bvec_strat:
        if grt.boffset > 0:
            if bvec_strategy_type.FMAIDX == layout.bvec_strat:
                asmblock += asm.add_greg_voff(asm.greg(grt.bareg), grt.boffset, datatype)
            else:
                asmblock += asm.add_greg_imm(asm.greg(grt.bareg), grt.boffset)
            grt.boffset = 0
    if preload_b > nr and short_b_offset > 0:
        asmblock += asm.jump("load_b_end")
        asmblock += asm.label("short_b_fixup")
        if bvec_strategy_type.FMAIDX == layout.bvec_strat:
            asmblock += asm.add_greg_voff(asm.greg(grt.bareg), short_b_offset, datatype)
        else:
            asmblock += asm.add_greg_imm(asm.greg(grt.bareg), short_b_offset)
        asmblock += asm.jump("k1novecload")
    asmblock += asm.label("load_b_end")
    rt.unuse_greg(tmpreg)
    return asmblock

# ================================#
# initialize vector registers for #
# storing/accumulation microtile  #
# ================================#

def vectorinit(asm : asmgen, grt : gemm_tracker,
               datatype : asm_data_type):
    first_vreg = grt.cvreg_first
    asmblock = ""

    # RVV: vsetvlmax for the datatype
    if asm.__class__.__name__.startswith("rvv"):
        sparereg_idx = grt.rt.reserve_any_greg()
        sparereg = asm.greg(sparereg_idx)
        asmblock += asm.vsetvlmax(sparereg, datatype)
        rt.unuse_greg(sparereg_idx)

    for i in range(grt.cvreg_count):
        asmblock += asm.zero_vreg(asm.vreg(i+first_vreg),datatype)
    return asmblock

def nanogemm(asm : asmgen, pf : prefetch_options,
             layout : kernel_layout, 
             vectors_in_mr : int, nr : int,
             unroll_factor : int, max_vregs : int,
             mem_use : mem_use_type, datatype : asm_data_type,
             params):
    """ Generate inline assembly for the inner part of a gemm microkernel
       
        The generated assembly block has to be placed in a __asm__ volatile ( <ASMBLOCK> ); statement in c/c++ code.
        This implies compatibility with the GCC inline assembly syntax. 
        The block includes the inputs, outputs and clobber lists.

        Parameters
        ----------
        asm : subclass(asmgen)
              Assembly generator. Must inherit from asmgen and implement it's abstract methods
        pf  : prefetch
              structure specifying the prefetch strategy
        layout : kernel_layout
                 structure specifying the kernel layout
        vectors_in_mr : uint
             m_r dimension of the microkernel in number of vector registers
        nr : uint
             n_r dimension of the microkernel in elements
        unroll_factor : uint
                        how many times to unroll the inner loop. Constraint: 0 == nr*unroll_factor % (#vregs for B)
        max_vregs: uint
                   max. number of vector register to use. Will use the first max_vregs vector registers.
        mem_use: mem_use_type
                 Specifies what kind of memory usage pattern to generate. See the definition of the enum.
        datatype: asm_data_type
                  Element data type (i.e double precision, single precision, ...)
        params: gemm_params
                  quirks and parameters specifying different options for generating the gemm kernel

    """

    rt = reg_tracker(max_greg=asm.max_gregs,
                     max_vreg=max_vregs,
                     max_freg=asm.max_fregs)

    bfreg_count = -1
    if bvec_strategy_type.FMAVF == layout.bvec_strat:
        bfreg_count = 2*nr if 2*nr < asm.max_fregs else nr
        #bfreg_count = asm.max_fregs()

    grt = gemm_tracker(rt=rt, 
                       avec_strat=layout.avec_strat, 
                       vec_in_mr=vectors_in_mr, nr=nr, 
                       bfreg_count=bfreg_count, 
                       bvreg_count = -1, 
                       aareg = 0, bareg = 1, careg = 2)

    bregs = grt.get_bvreg_count()
    if bvec_strategy_type.FMAVF == layout.bvec_strat:
        bregs = grt.get_bfreg_count()

    asmblock = vectorinit(asm=asm, grt=grt, datatype=datatype)
    bregs = min(nr*unroll_factor, bregs)
    asmblock += memoryinit(asm=asm, grt=grt, layout=layout,
                          nr=nr,
                          mem_use=mem_use, prefetch=pf,
                          preload_a=vectors_in_mr,
                          preload_b=bregs, datatype=datatype)
    

    loopreg = rt.reserve_any_greg()
    asmblock += asm.mov_param_to_greg("iterations", asm.greg(loopreg))
    asmblock += asm.jzero(asm.greg(loopreg), "kloopend")
    asmblock += asm.add_greg_imm(asm.greg(loopreg),-1)
    asmblock += asm.loopbegin_nz(asm.greg(loopreg),"kloop","klast")
    # TODO: layouts,instruction mixes, uarch-dependent methods (FMA into memory for avx....)

    grt.avreg_offset = 0
    grt.aoffset = 0
    grt.boffset = 0
    grt.cur_bvreg = 0
    grt.cur_bfreg = 0
    # Right now:
    # the last mr/simd_size * nr vregs are for the accumulation tile
    # The first vectors_in_mr*2 vectors are for A, rotating through them
    for _ in range(unroll_factor):
        asmblock += inner_kernel(asm=asm, grt=grt, layout=layout,
                                 mem_use=mem_use,
                                 vectors_in_mr=vectors_in_mr,nr=nr,
                                 no_unroll=False,
                                 unroll_factor=unroll_factor,
                                 wrapup_a=False,
                                 wrapup_b=False,
                                 datatype=datatype)
    asmblock += finalize_mm(asm=asm, grt=grt, layout=layout,
                            unroll_factor=unroll_factor,
                            nr=nr,
                            datatype=datatype)
    # reset pointers
    if mem_use_type.L1 == mem_use:
        asmblock += asm.mov_greg(asm.greg(grt.oldareg),asm.greg(grt.aareg))
        asmblock += asm.mov_greg(asm.greg(grt.oldbreg),asm.greg(grt.bareg))
    asmblock += asm.loopend(asm.greg(loopreg),"kloop")
    asmblock += asm.label("klast")
    for i in range(unroll_factor):
        asmblock += inner_kernel(asm=asm, grt=grt, layout=layout,
                                 mem_use=mem_use,
                                 vectors_in_mr=vectors_in_mr,nr=nr,
                                 no_unroll=False,
                                 unroll_factor=unroll_factor,
                                 wrapup_a=(i==(unroll_factor-1)),
                                 wrapup_b=True,
                                 datatype=datatype)
    asmblock += finalize_mm(asm=asm, grt=grt, layout=layout,
                            unroll_factor=unroll_factor,
                            nr=nr,
                            datatype=datatype)
    asmblock += asm.label("kloopend")


    asmblock += asm.mov_param_to_greg("kleft", asm.greg(loopreg))
    asmblock += asm.jzero(asm.greg(loopreg), "k1loopend")
    
    # We need to ensure that a and b are available for the 1xk loop 

    # preloading a
    offset = 0
    for i in range(vectors_in_mr):
        asmblock += asm.load_vector_voff(
                asm.greg(grt.aareg),
                offset,
                asm.vreg(grt.avreg(i)),
                datatype)
        offset += 1
        if offset > asm.max_load_voff:
            asmblock += add_voff(asm, asm.greg(grt.aareg), offset, 
                                 asm.greg(grt.vlreg), datatype)
            offset = 0

    if 0 != offset:
        asmblock += add_voff(asm, asm.greg(grt.aareg), offset, 
                             asm.greg(grt.vlreg), datatype)
        offset = 0

    # preloading b
    short_b_offset = 0
    elements_in_vector = asm.simd_size//datatype.value
    tmpreg = rt.reserve_any_greg()
    preload_b = nr
    if bvec_strategy_type.FMAIDX == layout.bvec_strat:
        preload_b = preload_b//elements_in_vector
    for i in range(preload_b):
        asmblock += load_b(asm=asm, grt=grt, layout=layout,
                           mem_use=mem_use, datatype=datatype)

        if bvec_strategy_type.FMAIDX == layout.bvec_strat:
            grt.cur_bfreg += elements_in_vector
            grt.cur_bvreg += elements_in_vector
        else:
            grt.cur_bfreg += 1
            grt.cur_bvreg += 1

    if bvec_strategy_type.NOLOAD != layout.bvec_strat:
        if grt.boffset > 0:
            if bvec_strategy_type.FMAIDX == layout.bvec_strat:
                asmblock += asm.add_greg_voff(asm.greg(grt.bareg), grt.boffset, datatype)
            else:
                asmblock += asm.add_greg_imm(asm.greg(grt.bareg), grt.boffset)
            grt.boffset = 0
    # In case the unrolled loop was skipped, a and b are preloaded
    # So put a label to jump to 
    asmblock += asm.label("k1novecload")
    # reset a and b vecs to start 
    grt.cur_avreg = 0
    grt.cur_bvreg = 0
    grt.cur_bfreg = 0
    grt.avreg_offset = 0

    asmblock += asm.add_greg_imm(asm.greg(loopreg),-1)
    asmblock += asm.loopbegin_nz(asm.greg(loopreg),"k1loop","k1last")
    asmblock += inner_kernel(asm=asm, grt=grt, layout=layout,
                             mem_use=mem_use,
                             vectors_in_mr=vectors_in_mr,nr=nr,
                             no_unroll=True,
                             unroll_factor=1,
                             wrapup_a=False,
                             wrapup_b=False,
                             datatype=datatype)
    asmblock += finalize_mm(asm=asm, grt=grt, layout=layout,
                            unroll_factor=1,
                            nr=nr,
                            datatype=datatype)
    if mem_use_type.L1 == mem_use:
        asmblock += asm.mov_greg(asm.greg(grt.oldareg),asm.greg(grt.aareg))
        asmblock += asm.mov_greg(asm.greg(grt.oldbreg),asm.greg(grt.bareg))
    asmblock += asm.loopend(asm.greg(loopreg),"k1loop")
    asmblock += asm.label("k1last")
    grt.cur_avreg = 0
    grt.cur_bvreg = 0
    grt.cur_bfreg = 0
    grt.avreg_offset = 0
    asmblock += inner_kernel(asm=asm, grt=grt, layout=layout,
                             mem_use=mem_use,
                             vectors_in_mr=vectors_in_mr,nr=nr,
                             no_unroll=True,
                             unroll_factor=1,
                             wrapup_a=True,
                             wrapup_b=True,
                             datatype=datatype)
    asmblock += asm.label("k1loopend")
    # We no longer need a and b address regs, so reuse them for alpha/beta
    asmblock += asm.load_pointer(asm.greg(grt.aareg), "alpha")
    asmblock += asm.load_pointer(asm.greg(grt.bareg), "beta")
    alphafreg = rt.reserve_any_freg()
    betafreg = rt.reserve_any_freg()

    # All avregs and bvregs are free now
    free_vregs=[bvreg for bvreg in range(grt.bvreg_first,grt.bvreg_first+grt.bvreg_count)]
    free_vregs+=[avreg for avreg in range(grt.avreg_first,grt.avreg_first+grt.avreg_count)]

    asmblock += asm.load_scalar_immoff(asm.greg(grt.aareg),
                                      0, 
                                      asm.freg(alphafreg),
                                      datatype)
    asmblock += asm.load_scalar_immoff(asm.greg(grt.bareg),
                                       0, 
                                       asm.freg(betafreg),
                                       datatype)

    tmpfreg = rt.reserve_any_freg()
    tmpgreg = rt.reserve_any_greg()
    asmblock += asm.jfzero(asm.freg(betafreg), asm.freg(tmpfreg), asm.greg(tmpgreg), "beta0", datatype)
    rt.unuse_freg(tmpfreg)
    rt.unuse_greg(tmpgreg)

    alphareg = 0
    betareg = 0
    if bvec_strategy_type.FMAVF == layout.bvec_strat:
        alphareg = alphafreg
        betareg = betafreg
    else:
        # TODO: Check for possibly running out of vector registers
        alphavreg = free_vregs.pop()
        betavreg = free_vregs.pop()
        alphareg = alphavreg
        betareg = betavreg
        asmblock += asm.load_vector_dist1(asm.greg(grt.aareg),
                                          0, 
                                          asm.vreg(alphavreg),
                                          datatype)
        asmblock += asm.load_vector_dist1(asm.greg(grt.bareg),
                                          0, 
                                          asm.vreg(betavreg),
                                          datatype)
    asmblock += update_c_tile(asm, rt, grt,
                              layout, mem_use,
                              free_vregs,
                              alphareg,
                              betareg,
                              vectors_in_mr, nr,
                              False,
                              datatype)
    asmblock += asm.jump("beta0end")
    asmblock += asm.label("beta0")

    # reset free vregs between beta nonzero and beta zero versions (removing accum tile vregs)
    free_vregs=[bvreg for bvreg in range(grt.bvreg_first,grt.bvreg_first+grt.bvreg_count)]
    free_vregs+=[avreg for avreg in range(grt.avreg_first,grt.avreg_first+grt.avreg_count)]

    alphareg = 0
    betareg = 0
    if bvec_strategy_type.FMAVF == layout.bvec_strat:
        alphareg = alphafreg
    else:
        # TODO: Check for possibly running out of vector registers
        alphavreg = free_vregs.pop()
        alphareg = alphavreg
        asmblock += asm.load_vector_dist1(asm.greg(grt.aareg),
                                          0, 
                                          asm.vreg(alphavreg),
                                          datatype)
    asmblock += update_c_tile(asm, rt, grt,
                              layout, mem_use,
                              free_vregs,
                              alphareg,
                              betareg,
                              vectors_in_mr, nr,
                              True,
                              datatype)
    asmblock += asm.label("beta0end")
    #asmblock += asm.label("alpha1betazero")
    #asmblock += asm.label("alpha1")

    if asm.__class__.__name__.startswith("rvv"):
        rt.unuse_greg(grt.vlreg)
    if mem_use_type.L1 == mem_use:
        rt.unuse_greg(grt.oldareg)
        rt.unuse_greg(grt.oldbreg)

    clobber_vregs = [asm.vreg(i) for i in range(max_vregs)]
    # one greg for counter, one for A address, one for B address
    clobber_gregs = [asm.greg(i) for i in rt.get_clobbered_gregs()]
    inputs = []
    outputs = []
    # Need to indicate to compiler that we'll write to the memory pointed at by c
    outputs.append(('dummy_c', '+m', f"*({c_data_types[datatype]} (*)[]) c"))
    # Not sure if required, but clang can't handle these
    #inputs.append(('dummy_a', 'm', f"*({c_data_types[datatype]} (*)[]) a"))
    #inputs.append(('dummy_b', 'm', f"*({c_data_types[datatype]} (*)[]) b"))
    inputs.append(('iterations','m','(iterations)'))
    inputs.append(('kleft','m','(kleft)'))
    inputs.append(('a','m','(a)'))
    inputs.append(('b','m','(b)'))
    inputs.append(('c','m','(c)'))
    inputs.append(('alpha','m','(alpha)'))
    inputs.append(('beta','m','(beta)'))
    asmblock += asm.operands(inputs=inputs,outputs=outputs,clobber=clobber_gregs+clobber_vregs)

    return asmblock
