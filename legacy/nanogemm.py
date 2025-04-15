from asmgen.asmblocks.noarch import asmgen
from asmgen.registers import(
        greg_base, freg_base, vreg_base,
        asm_data_type as adt,
        adt_size,
        reg_tracker)
from asmgen.asmblocks.operations import modifier as mod

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
        self.rt.reserve_specific_reg('greg',aareg)
        self.rt.reserve_specific_reg('greg',bareg)
        self.rt.reserve_specific_reg('greg',careg)

        self.cvreg_count = vec_in_mr*nr
        self.cvreg_first = self.rt.max_regs['vreg'] - self.cvreg_count
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

        assert all(v < self.rt.max_regs['vreg'] for v in vreg_range), f"Some vector registers chosen over max vec. register ({self.rt.max_regs['vreg']})"

        for vreg in vreg_range:
            self.rt.reserve_specific_reg('vreg',vreg)

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
            return offset+adt_size(dt)
        elif mem_use == mem_use_type.CONTIGUOUS:
            return offset+adt_size(dt)
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

def add_voff(asm : asmgen, areg : greg_base, aoffset, vlreg, dt):
    asmblock = ""
    if asm.max_add_voff > 0:
        asmblock += asm.add_greg_voff(reg=areg, offset=aoffset, dt=dt)
    else:
        asmblock += asm.add_greg_greg(dst=areg, reg1=areg, reg2=vlreg)
    return asmblock

# ============================== #
# loading 1 vector from A matrix #
# ============================== #

def load_a_vec(asm : asmgen, grt : gemm_tracker,
               mem_use : mem_use_type, dt : adt):
    asmblock  = asm.load_vector_voff(areg=asm.greg(grt.aareg), 
                                     voffset=grt.aoffset, 
                                     vreg=asm.vreg(grt.cur_avreg),
                                     dt=dt)
    grt.aoffset = advance_vecaddr_voffset(grt.aoffset, mem_use)
    # There is a limit to immediate offsets, i.e for SVE you can offset -8 to 7 vectors
    # if we roll over this limit, add the offset to the a address register and 
    if grt.aoffset > asm.max_load_voff:
        asmblock += add_voff(asm, asm.greg(grt.aareg), grt.aoffset,
                             asm.greg(grt.vlreg), dt)
        grt.aoffset = 0
    return asmblock

# ============================= #
# loading 1 value from B matrix #
# ============================= #

def load_b(asm : asmgen, grt : gemm_tracker,
           layout : kernel_layout,
           mem_use : mem_use_type, dt : adt):
    asmblock = ""
    boffcompare = 0
    boffadd = lambda reg,off : ""

    elements_in_vector = asm.simd_size//adt_size(dt)


    if bvec_strategy_type.DIST1_INC == layout.bvec_strat:
        asmblock += asm.load_vector_dist1_inc(areg=asm.greg(grt.bareg), 
                                              offset=adt_size(dt), 
                                              vreg=asm.vreg(grt.bvreg_rot(grt.cur_bvreg)),
                                              dt=dt)
        # Not important as offset stays
        boffcompare = asm.max_load_immoff(dt)
        boffadd = lambda greg, boffset : asm.add_greg_imm(reg=greg, imm=boffset)
    elif bvec_strategy_type.DIST1_BOFF == layout.bvec_strat:
        asmblock += asm.load_vector_dist1_boff(areg=asm.greg(grt.bareg),
                                               offset=grt.boffset, 
                                               vreg=asm.vreg(grt.bvreg_rot(grt.cur_bvreg)),
                                               dt=dt)
        boffcompare = asm.max_load_immoff(dt)
        boffadd = lambda greg, boffset : asm.add_greg_imm(reg=greg, imm=boffset)
    elif bvec_strategy_type.FMAIDX == layout.bvec_strat:
        asmblock += asm.load_vector_voff(areg=asm.greg(grt.bareg),
                                         voffset=grt.boffset, 
                                         vreg=asm.vreg(grt.bvreg_rot(grt.cur_bvreg//elements_in_vector)),
                                         dt=dt)
        boffcompare = asm.max_load_voff
        boffadd = lambda greg, voffset : asm.add_greg_voff(reg=greg, offset=voffset, dt=dt)
    elif bvec_strategy_type.FMAVF == layout.bvec_strat:
        asmblock += asm.load_scalar_immoff(areg=asm.greg(grt.bareg),
                                           offset=grt.boffset, 
                                           freg=asm.freg(grt.bfreg_rot(grt.cur_bfreg),dt=dt),
                                           dt=dt)
        boffcompare = asm.max_fload_immoff(dt)
        boffadd = lambda greg, boffset : asm.add_greg_imm(reg=greg, imm=boffset)

    if bvec_strategy_type.NOLOAD != layout.bvec_strat:
        grt.boffset = advance_b_offset(grt.boffset, asm, layout.bvec_strat, mem_use, dt)
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
                  dt):
    c_tile_store_queue : list[int] = []
    coffset=0
    cur_c_m=0
    cur_c_m_s=0
    csoffset=0
    # register holding address for storing C tile
    casreg = rt.reserve_any_reg('greg')
    # copy from c source reg
    asmblock = asm.mov_greg(src=asm.greg(grt.careg), dst=asm.greg(casreg))
    cscreg = rt.reserve_any_reg('greg')
    asmblock += asm.mov_param_to_greg(param="cs_c",dst=asm.greg(cscreg))
    asmblock += asm.shift_greg_left(reg=asm.greg(cscreg),bit_count=adt_size(dt).bit_length()-1)

    # TODO: this is crap, also written only with RVV in mind
    if asm.max_load_voff < vectors_in_mr:
        # TODO: multiply instead of this
        asmblock += "".join([
            asm.sub_greg_greg(dst=asm.greg(cscreg), reg1=asm.greg(cscreg), reg2=asm.greg(grt.vlreg)) for i in range(vectors_in_mr-asm.max_load_voff-1)
        ])
    # TODO: Deduplicate code for c store in and after loop
    # TODO: Deduplicate code for advancing offsets (also in the code above)
    for i in range(nr):
        for j in range(vectors_in_mr):
            tile_idx = grt.cvreg(i*vectors_in_mr+j)

            # Start storing if we run out of free regs or already halfway through
            if 0 == len(free_vregs) or len(c_tile_store_queue) > ((nr*vectors_in_mr)//2):
                csvreg = c_tile_store_queue.pop(0)
                asmblock += asm.store_vector_voff(areg=asm.greg(casreg),
                                      voffset=csoffset,
                                      vreg=asm.vreg(csvreg),
                                      dt=dt)
                free_vregs.append(csvreg)
                csoffset = advance_vecaddr_voffset(csoffset, mem_use)
                cur_c_m_s += 1
                if ((0 == (cur_c_m_s % vectors_in_mr)) and not ((layout.bvec_strat == bvec_strategy_type.NOLOAD) or (mem_use == mem_use_type.SAMEDATA))):
                    asmblock += asm.add_greg_greg(dst=asm.greg(casreg), reg1=asm.greg(casreg), reg2=asm.greg(cscreg))
                    cur_c_m_s = 0
                    csoffset = 0
                elif not ((layout.bvec_strat == bvec_strategy_type.NOLOAD) or (mem_use == mem_use_type.SAMEDATA) or (0 != (cur_c_m_s % vectors_in_mr))):
                    raise NotImplementedError("Special case not handled yet (mr larger than max. vector offset expressable in asm)")
                if csoffset > asm.max_load_voff:
                    asmblock += add_voff(asm, asm.greg(casreg), csoffset, 
                                         asm.greg(grt.vlreg), dt)
                    csoffset = 0


            cvreg = 0
            # get a free vreg from the beginning of the list
            if not beta0:
                cvreg = free_vregs.pop(0)
                asmblock += asm.load_vector_voff(areg=asm.greg(grt.careg), 
                                                 voffset=coffset, 
                                                 vreg=asm.vreg(cvreg),
                                                 dt=dt)
                coffset = advance_vecaddr_voffset(coffset, mem_use)
                cur_c_m += 1
                if ((0 == (cur_c_m % vectors_in_mr)) and not ((layout.bvec_strat == bvec_strategy_type.NOLOAD) or (mem_use == mem_use_type.SAMEDATA))):
                    asmblock += asm.add_greg_greg(dst=asm.greg(grt.careg), reg1=asm.greg(grt.careg), reg2=asm.greg(cscreg))
                    cur_c_m = 0
                elif not ((layout.bvec_strat == bvec_strategy_type.NOLOAD) or (mem_use == mem_use_type.SAMEDATA) or (0 != (cur_c_m % vectors_in_mr))):
                    raise NotImplementedError(f"Special case not handled yet (mr larger than max. vector offset expressable in asm):\n coffset={coffset}\n vectors_in_mr={vectors_in_mr}\n max_load_voff={asm.max_load_voff}")
                if coffset > asm.max_load_voff:
                    asmblock += add_voff(asm, asm.greg(grt.careg), coffset, 
                                         asm.greg(grt.vlreg), dt)
                    coffset = 0

            if layout.bvec_strat == bvec_strategy_type.FMAVF:
                if not beta0:
                    asmblock += asm.fmul(adreg=asm.vreg(cvreg),
                                         bdreg=asm.freg(betareg,dt=dt),
                                         cdreg=asm.vreg(cvreg),
                                         a_dt=dt, b_dt=dt, c_dt=dt,
                                         modifiers={mod.VF})
                    asmblock += asm.fma(adreg=asm.vreg(tile_idx),
                                        bdreg=asm.freg(alphareg,dt=dt),
                                        cdreg=asm.vreg(cvreg),
                                        a_dt=dt, b_dt=dt, c_dt=dt,
                                        modifiers={mod.VF})
                    c_tile_store_queue.append(cvreg)
                    free_vregs.append(tile_idx)
                else:
                    asmblock += asm.fmul(adreg=asm.vreg(tile_idx),
                                         bdreg=asm.freg(alphareg,dt=dt),
                                         cdreg=asm.vreg(tile_idx),
                                         a_dt=dt, b_dt=dt, c_dt=dt,
                                         modifiers={mod.VF})
                    c_tile_store_queue.append(tile_idx)
            else:
                if not beta0:
                    asmblock += asm.fmul(adreg=asm.vreg(cvreg),
                                         bdreg=asm.vreg(betareg),
                                         cdreg=asm.vreg(cvreg),
                                         a_dt=dt, b_dt=dt, c_dt=dt)
                    asmblock += asm.fma(adreg=asm.vreg(alphareg),
                                        bdreg=asm.vreg(tile_idx),
                                        cdreg=asm.vreg(cvreg),
                                        a_dt=dt, b_dt=dt, c_dt=dt)
                    c_tile_store_queue.append(cvreg)
                    free_vregs.append(tile_idx)
                else:
                    asmblock += asm.fmul(adreg=asm.vreg(tile_idx),
                                         bdreg=asm.vreg(alphareg),
                                         cdreg=asm.vreg(tile_idx),
                                         a_dt=dt, b_dt=dt, c_dt=dt)
                    c_tile_store_queue.append(tile_idx)
            # TODO: can tile_idx == cvreg happen? (would already cause trouble before, 
            #       but check anyways)
            
            # Alternatively:
            # asmblock += asm.fmul(asm.vreg(tile_idx),
            #                      asm.vreg(alphareg),
            #                      asm.vreg(tile_idx),
            #                      dt)
            # asmblock += asm.fma(asm.vreg(betareg),
            #                     asm.vreg(cvreg),
            #                     asm.vreg(tile_idx),
            #                      dt)
            # c_tile_store_queue.append(tile_idx)
            # free_vregs.append(cvreg)
        #TODO: inc column here

    for csvreg in c_tile_store_queue:
        asmblock += asm.store_vector_voff(
                              areg=asm.greg(casreg),
                              voffset=csoffset,
                              vreg=asm.vreg(csvreg),
                              dt=dt)
        csoffset = advance_vecaddr_voffset(csoffset, mem_use)
        cur_c_m_s += 1
        if ((0 == (cur_c_m_s % vectors_in_mr)) and not ((layout.bvec_strat == bvec_strategy_type.NOLOAD) or (mem_use == mem_use_type.SAMEDATA))):
            asmblock += asm.add_greg_greg(dst=asm.greg(casreg), reg1=asm.greg(casreg), reg2=asm.greg(cscreg))
            cur_c_m_s = 0
            csoffset = 0
        elif not ((layout.bvec_strat == bvec_strategy_type.NOLOAD) or (mem_use == mem_use_type.SAMEDATA) or (0 != (cur_c_m_s % vectors_in_mr))):
            raise NotImplementedError("Special case not handled yet (mr larger than max. vector offset expressable in asm)")
        if csoffset > asm.max_load_voff:
            asmblock += add_voff(asm, asm.greg(casreg), csoffset, 
                                 asm.greg(grt.vlreg), dt)
            csoffset = 0
    rt.unuse_reg('greg', casreg)

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
                 dt : adt):
    asmblock = ""
    ignore_a_load = wrapup_a
    elements_in_vector = asm.simd_size//adt_size(dt)
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
                                           mem_use, dt)
            # FMA HERE
            avreg_id = (grt.avreg_offset+j)
            if layout.bvec_strat in [bvec_strategy_type.DIST1_BOFF,bvec_strategy_type.DIST1_INC,bvec_strategy_type.NOLOAD]:
                asmblock += asm.fma(adreg=asm.vreg(grt.avreg(avreg_id)),
                                    bdreg=asm.vreg(grt.bvreg_rot(grt.cur_bvreg)), 
                                    cdreg=asm.vreg(tile_idx),
                                    a_dt=dt, b_dt=dt, c_dt=dt)
            elif bvec_strategy_type.FMAIDX == layout.bvec_strat:
                asmblock += asm.fma(adreg=asm.vreg(grt.avreg(avreg_id)),
                                    bdreg=asm.vreg(grt.bvreg_rot(grt.cur_bvreg//elements_in_vector)),
                                    cdreg=asm.vreg(tile_idx),
                                    a_dt=dt, b_dt=dt, c_dt=dt,
                                    modifiers={mod.IDX}, idx=grt.cur_bvreg % elements_in_vector)
            elif bvec_strategy_type.FMAVF == layout.bvec_strat:
                asmblock += asm.fma(adreg=asm.vreg(grt.avreg(avreg_id)),
                                    bdreg=asm.freg(grt.bfreg_rot(grt.cur_bfreg), dt=dt),
                                    cdreg=asm.vreg(tile_idx),
                                    a_dt=dt, b_dt=dt, c_dt=dt,
                                    modifiers={mod.VF})

        if bvec_strategy_type.FMAVF == layout.bvec_strat:
            ignore_b_load = wrapup_b and (grt.cur_bfreg > unroll_factor*nr-grt.get_bfreg_count()-1)
        else:
            ignore_b_load = wrapup_b and (grt.cur_bvreg > unroll_factor*nr-grt.get_bvreg_count()-1)

        if not ignore_b_load:
            asmblock += load_b(asm=asm, grt=grt, layout=layout,
                               mem_use=mem_use, dt=dt)
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
                asmblock += load_a_vec(asm, grt, mem_use, dt)

    return asmblock

# ================================= #
# clean up after k-loop             #
# (finalize matrix-matrix multiply) #
# ================================= #

def finalize_mm(asm : asmgen, grt :gemm_tracker, 
                layout : kernel_layout, unroll_factor : int,
                nr : int,
                dt : adt) -> str:
    asmblock = ""
    bvreg_rem = 0
    reg_count = grt.get_bvreg_count()
    elements_in_vector = asm.simd_size//adt_size(dt)
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
                             asm.greg(grt.vlreg), dt)
        grt.aoffset = 0
    if grt.boffset > 0:
        asmblock += asm.add_greg_imm(reg=asm.greg(grt.bareg), imm=grt.boffset)
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
               dt : adt):
    assert isinstance(layout, kernel_layout), f"Not a kernel layout: {layout}"
    assert isinstance(dt, adt), f"Not an adt: {dt}"

    rt = grt.rt
    asmblock  = asm.load_pointer(areg=asm.greg(grt.aareg), name="a")
    asmblock += asm.load_pointer(areg=asm.greg(grt.bareg), name="b")
    asmblock += asm.load_pointer(areg=asm.greg(grt.careg), name="c")

    if mem_use_type.L1 == mem_use:
        oldareg_idx = rt.reserve_any_reg('greg')
        oldbreg_idx = rt.reserve_any_reg('greg')
        oldareg = asm.greg(oldareg_idx)
        oldbreg = asm.greg(oldbreg_idx)
        grt.oldareg = oldareg_idx
        grt.oldbreg = oldbreg_idx
        asmblock += asm.mov_greg(src=asm.greg(grt.aareg), dst=oldareg)
        asmblock += asm.mov_greg(src=asm.greg(grt.bareg), dst=oldbreg)

    # ================================= #
    #             ISA QUIRKS            #
    # ================================= #

    # SVE: set p0 to true
    if asm.__class__.__name__ == "sve":
        asmblock += asm.ptrue(asm.preg(0),dt)

    # RVV: vsetvlmax for the dt
    if asm.__class__.__name__.startswith("rvv"):
        sparereg_idx = rt.reserve_any_reg('greg')
        sparereg = asm.greg(sparereg_idx)
        asmblock += asm.vsetvlmax(reg=sparereg, dt=dt)
        rt.unuse_reg('greg', sparereg_idx)

    if asm.__class__.__name__.startswith("rvv"):
        vlreg_idx = rt.reserve_any_reg('greg')
        vlreg = asm.greg(vlreg_idx)
        asmblock += asm.simd_size_to_greg(reg=vlreg, dt=dt)
        asmblock += asm.shift_greg_left(reg=vlreg,bit_count=adt_size(dt).bit_length()-1)
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
        tmpreg = rt.reserve_any_reg('greg')
        for _ in range(count):
            asmblock += asm.prefetch_l1_boff(areg=asm.greg(areg), offset=offset)
            offset += prefetch.cl_size
            if offset > asm.max_prefetch_offset:
                if not restore_areg:
                    asmblock += asm.mov_greg(asm.greg(areg), asm.greg(tmpreg))
                asmblock += asm.add_greg_imm(asm.greg(areg), offset)
                restore_areg = True
        if restore_areg:
            asmblock += asm.mov_greg(asm.greg(tmpreg), asm.greg(areg))
        rt.unuse_reg('greg', tmpreg)

    # ================================= #
    #             PRELOADING            #
    # ================================= #


    # preloading a
    offset = 0
    for i in range(preload_a):
        asmblock += asm.load_vector_voff(
                areg=asm.greg(grt.aareg),
                voffset=offset,
                vreg=asm.vreg(grt.avreg(i)),
                dt=dt)
        offset += 1
        if offset > asm.max_load_voff:
            asmblock += add_voff(asm, asm.greg(grt.aareg), offset, 
                                 asm.greg(grt.vlreg), dt)
            offset = 0

    if 0 != offset:
        asmblock += add_voff(asm, asm.greg(grt.aareg), offset, 
                             asm.greg(grt.vlreg), dt)
        offset = 0

    # preloading b
    short_b_offset = 0
    elements_in_vector = asm.simd_size//adt_size(dt)
    tmpreg = rt.reserve_any_reg('greg')
    if bvec_strategy_type.FMAIDX == layout.bvec_strat:
        preload_b = preload_b//elements_in_vector
    for i in range(preload_b):
        asmblock += load_b(asm=asm, grt=grt, layout=layout,
                           mem_use=mem_use, dt=dt)

        if bvec_strategy_type.FMAIDX == layout.bvec_strat:
            grt.cur_bfreg += elements_in_vector
            grt.cur_bvreg += elements_in_vector
        else:
            grt.cur_bfreg += 1
            grt.cur_bvreg += 1

        if nr-1 == i and preload_b > nr and grt.boffset > 0:
            asmblock += asm.mov_param_to_greg(param="iterations",dst=asm.greg(tmpreg))
            asmblock += asm.jzero(reg=asm.greg(tmpreg), label="short_b_fixup")
            short_b_offset = grt.boffset

    if bvec_strategy_type.NOLOAD != layout.bvec_strat:
        if grt.boffset > 0:
            if bvec_strategy_type.FMAIDX == layout.bvec_strat:
                asmblock += asm.add_greg_voff(reg=asm.greg(grt.bareg), offset=grt.boffset, dt=dt)
            else:
                asmblock += asm.add_greg_imm(reg=asm.greg(grt.bareg), imm=grt.boffset)
            grt.boffset = 0
    if preload_b > nr and short_b_offset > 0:
        asmblock += asm.jump(label="load_b_end")
        asmblock += asm.label(label="short_b_fixup")
        if bvec_strategy_type.FMAIDX == layout.bvec_strat:
            asmblock += asm.add_greg_voff(asm.greg(grt.bareg), short_b_offset, dt)
        else:
            asmblock += asm.add_greg_imm(reg=asm.greg(grt.bareg), imm=short_b_offset)
        asmblock += asm.jump(label="k1novecload")
    asmblock += asm.label(label="load_b_end")
    rt.unuse_reg('greg', tmpreg)
    return asmblock

# ================================#
# initialize vector registers for #
# storing/accumulation microtile  #
# ================================#

def vectorinit(asm : asmgen, grt : gemm_tracker,
               dt : adt):
    first_vreg = grt.cvreg_first
    asmblock = ""

    # RVV: vsetvlmax for the dt
    if asm.__class__.__name__.startswith("rvv"):
        sparereg_idx = grt.rt.reserve_any_reg('greg')
        sparereg = asm.greg(sparereg_idx)
        asmblock += asm.vsetvlmax(reg=sparereg, dt=dt)
        grt.rt.unuse_reg('greg', sparereg_idx)

    for i in range(grt.cvreg_count):
        asmblock += asm.zero_vreg(vreg=asm.vreg(i+first_vreg),dt=dt)
    return asmblock

def nanogemm(asm : asmgen, pf : prefetch_options,
             layout : kernel_layout, 
             vectors_in_mr : int, nr : int,
             unroll_factor : int, max_vregs : int,
             mem_use : mem_use_type, dt : adt,
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
        dt: adt
                  Element data type (i.e double precision, single precision, ...)
        params: gemm_params
                  quirks and parameters specifying different options for generating the gemm kernel

    """

    rt = reg_tracker([
        ('greg',asm.max_gregs),
        ('vreg',max_vregs),
        ('freg',asm.max_fregs)])

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

    asmblock = vectorinit(asm=asm, grt=grt, dt=dt)
    bregs = min(nr*unroll_factor, bregs)
    asmblock += memoryinit(asm=asm, grt=grt, layout=layout,
                          nr=nr,
                          mem_use=mem_use, prefetch=pf,
                          preload_a=vectors_in_mr,
                          preload_b=bregs, dt=dt)
    

    loopreg = rt.reserve_any_reg('greg')
    asmblock += asm.mov_param_to_greg(param="iterations", dst=asm.greg(loopreg))
    asmblock += asm.jzero(reg=asm.greg(loopreg), label="kloopend")
    asmblock += asm.add_greg_imm(reg=asm.greg(loopreg),imm=-1)
    asmblock += asm.loopbegin_nz(reg=asm.greg(loopreg),label="kloop",labelskip="klast")
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
                                 dt=dt)
    asmblock += finalize_mm(asm=asm, grt=grt, layout=layout,
                            unroll_factor=unroll_factor,
                            nr=nr,
                            dt=dt)
    # reset pointers
    if mem_use_type.L1 == mem_use:
        asmblock += asm.mov_greg(src=asm.greg(grt.oldareg),dst=asm.greg(grt.aareg))
        asmblock += asm.mov_greg(src=asm.greg(grt.oldbreg),dst=asm.greg(grt.bareg))
    asmblock += asm.loopend(reg=asm.greg(loopreg),label="kloop")
    asmblock += asm.label(label="klast")
    for i in range(unroll_factor):
        asmblock += inner_kernel(asm=asm, grt=grt, layout=layout,
                                 mem_use=mem_use,
                                 vectors_in_mr=vectors_in_mr,nr=nr,
                                 no_unroll=False,
                                 unroll_factor=unroll_factor,
                                 wrapup_a=(i==(unroll_factor-1)),
                                 wrapup_b=True,
                                 dt=dt)
    asmblock += finalize_mm(asm=asm, grt=grt, layout=layout,
                            unroll_factor=unroll_factor,
                            nr=nr,
                            dt=dt)
    asmblock += asm.label(label="kloopend")


    asmblock += asm.mov_param_to_greg(param="kleft", dst=asm.greg(loopreg))
    asmblock += asm.jzero(reg=asm.greg(loopreg), label="k1loopend")
    
    # We need to ensure that a and b are available for the 1xk loop 

    # preloading a
    offset = 0
    for i in range(vectors_in_mr):
        asmblock += asm.load_vector_voff(
                areg=asm.greg(grt.aareg),
                voffset=offset,
                vreg=asm.vreg(grt.avreg(i)),
                dt=dt)
        offset += 1
        if offset > asm.max_load_voff:
            asmblock += add_voff(asm, asm.greg(grt.aareg), offset, 
                                 asm.greg(grt.vlreg), dt)
            offset = 0

    if 0 != offset:
        asmblock += add_voff(asm, asm.greg(grt.aareg), offset, 
                             asm.greg(grt.vlreg), dt)
        offset = 0

    # preloading b
    short_b_offset = 0
    elements_in_vector = asm.simd_size//adt_size(dt)
    tmpreg = rt.reserve_any_reg('greg')
    preload_b = nr
    if bvec_strategy_type.FMAIDX == layout.bvec_strat:
        preload_b = preload_b//elements_in_vector
    for i in range(preload_b):
        asmblock += load_b(asm=asm, grt=grt, layout=layout,
                           mem_use=mem_use, dt=dt)

        if bvec_strategy_type.FMAIDX == layout.bvec_strat:
            grt.cur_bfreg += elements_in_vector
            grt.cur_bvreg += elements_in_vector
        else:
            grt.cur_bfreg += 1
            grt.cur_bvreg += 1

    if bvec_strategy_type.NOLOAD != layout.bvec_strat:
        if grt.boffset > 0:
            if bvec_strategy_type.FMAIDX == layout.bvec_strat:
                asmblock += asm.add_greg_voff(reg=asm.greg(grt.bareg), offset=grt.boffset, dt=dt)
            else:
                asmblock += asm.add_greg_imm(reg=asm.greg(grt.bareg), imm=grt.boffset)
            grt.boffset = 0
    # In case the unrolled loop was skipped, a and b are preloaded
    # So put a label to jump to 
    asmblock += asm.label(label="k1novecload")
    # reset a and b vecs to start 
    grt.cur_avreg = 0
    grt.cur_bvreg = 0
    grt.cur_bfreg = 0
    grt.avreg_offset = 0

    asmblock += asm.add_greg_imm(reg=asm.greg(loopreg),imm=-1)
    asmblock += asm.loopbegin_nz(reg=asm.greg(loopreg),label="k1loop",labelskip="k1last")
    asmblock += inner_kernel(asm=asm, grt=grt, layout=layout,
                             mem_use=mem_use,
                             vectors_in_mr=vectors_in_mr,nr=nr,
                             no_unroll=True,
                             unroll_factor=1,
                             wrapup_a=False,
                             wrapup_b=False,
                             dt=dt)
    asmblock += finalize_mm(asm=asm, grt=grt, layout=layout,
                            unroll_factor=1,
                            nr=nr,
                            dt=dt)
    if mem_use_type.L1 == mem_use:
        asmblock += asm.mov_greg(src=asm.greg(grt.oldareg),dst=asm.greg(grt.aareg))
        asmblock += asm.mov_greg(src=asm.greg(grt.oldbreg),dst=asm.greg(grt.bareg))
    asmblock += asm.loopend(reg=asm.greg(loopreg),label="k1loop")
    asmblock += asm.label(label="k1last")
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
                             dt=dt)
    asmblock += asm.label(label="k1loopend")
    # We no longer need a and b address regs, so reuse them for alpha/beta
    asmblock += asm.load_pointer(areg=asm.greg(grt.aareg), name="alpha")
    asmblock += asm.load_pointer(areg=asm.greg(grt.bareg), name="beta")
    alphafreg = rt.reserve_any_reg('freg')
    betafreg = rt.reserve_any_reg('freg')

    # All avregs and bvregs are free now
    free_vregs=[bvreg for bvreg in range(grt.bvreg_first,grt.bvreg_first+grt.bvreg_count)]
    free_vregs+=[avreg for avreg in range(grt.avreg_first,grt.avreg_first+grt.avreg_count)]

    asmblock += asm.load_scalar_immoff(areg=asm.greg(grt.aareg),
                                      offset=0, 
                                      freg=asm.freg(alphafreg, dt=dt),
                                      dt=dt)
    asmblock += asm.load_scalar_immoff(areg=asm.greg(grt.bareg),
                                       offset=0, 
                                       freg=asm.freg(betafreg, dt=dt),
                                       dt=dt)

    tmpfreg = rt.reserve_any_reg('freg')
    tmpgreg = rt.reserve_any_reg('greg')
    asmblock += asm.jfzero(freg1=asm.freg(betafreg,dt=dt), freg2=asm.freg(tmpfreg,dt=dt), greg=asm.greg(tmpgreg), label="beta0", dt=dt)
    rt.unuse_reg('freg', tmpfreg)
    rt.unuse_reg('greg', tmpgreg)

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
        asmblock += asm.load_vector_dist1(areg=asm.greg(grt.aareg),
                                          vreg=asm.vreg(alphavreg),
                                          dt=dt)
        asmblock += asm.load_vector_dist1(areg=asm.greg(grt.bareg),
                                          vreg=asm.vreg(betavreg),
                                          dt=dt)
    asmblock += update_c_tile(asm, rt, grt,
                              layout, mem_use,
                              free_vregs,
                              alphareg,
                              betareg,
                              vectors_in_mr, nr,
                              False,
                              dt)
    asmblock += asm.jump(label="beta0end")
    asmblock += asm.label(label="beta0")

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
        asmblock += asm.load_vector_dist1(areg=asm.greg(grt.aareg),
                                          vreg=asm.vreg(alphavreg),
                                          dt=dt)
    asmblock += update_c_tile(asm, rt, grt,
                              layout, mem_use,
                              free_vregs,
                              alphareg,
                              betareg,
                              vectors_in_mr, nr,
                              True,
                              dt)
    asmblock += asm.label(label="beta0end")
    #asmblock += asm.label("alpha1betazero")
    #asmblock += asm.label("alpha1")

    if asm.__class__.__name__.startswith("rvv"):
        rt.unuse_reg('greg', grt.vlreg)
    if mem_use_type.L1 == mem_use:
        rt.unuse_reg('greg', grt.oldareg)
        rt.unuse_reg('greg', grt.oldbreg)

    clobber_vregs = [asm.vreg(i) for i in range(max_vregs)]
    clobber_fregs = [asm.freg(i,dt=dt) for i in range(asm.max_fregs)]
    # one greg for counter, one for A address, one for B address
    clobber_gregs = [asm.greg(i) for i in rt.get_clobbered_regs('greg')]
    inputs = []
    outputs = []
    # Need to indicate to compiler that we'll write to the memory pointed at by c
    outputs.append(('dummy_c', '+m', f"*({c_data_types[dt]} (*)[]) c"))
    # Not sure if required, but clang can't handle these
    #inputs.append(('dummy_a', 'm', f"*({c_data_types[dt]} (*)[]) a"))
    #inputs.append(('dummy_b', 'm', f"*({c_data_types[dt]} (*)[]) b"))
    inputs.append(('iterations','m','(iterations)'))
    inputs.append(('cs_c','m','(cs_c)'))
    inputs.append(('kleft','m','(kleft)'))
    inputs.append(('a','m','(a)'))
    inputs.append(('b','m','(b)'))
    inputs.append(('c','m','(c)'))
    inputs.append(('alpha','m','(alpha)'))
    inputs.append(('beta','m','(beta)'))
    asmblock += asm.operands(inputs=inputs,outputs=outputs,clobber=clobber_gregs+clobber_vregs+clobber_fregs)

    return asmblock
