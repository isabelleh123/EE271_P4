#ifndef __PROCESSING_ELEMENT_H__
#define __PROCESSING_ELEMENT_H__

#pragma once

#include <systemc.h>
#include <nvhls_connections.h>
#include <nvhls_int.h>
#include "../Spec.h"
#include "../InstructionSpec.h"

using namespace nvhls;

SC_MODULE(ProcessingElement)
{
    sc_in<bool> clk;
    sc_in<bool> rst;

    Connections::In<spec::Instruction::ProcessingElementInstruction::Packed_t> pe_inst_in;
    Connections::In<spec::VectorType> vector_input;
    Connections::In<spec::VectorType> matrix_input;
    Connections::Out<spec::VectorType> vector_output;

    spec::AccumVectorType accumulation_register;


    SC_HAS_PROCESS(ProcessingElement);
    ProcessingElement(sc_module_name name)
        : sc_module(name),
          clk("clk"),
          rst("rst"),
          pe_inst_in("pe_inst_in"),
          vector_input("vector_input"),
          matrix_input("matrix_input"),
          vector_output("vector_output")

    {
        SC_THREAD(run);
        sensitive << clk.pos();
        async_reset_signal_is(rst, false);
    }

    // Executes one non-CLR instruction on every lane of the accumulator.
    //   W     : width in bits of one input/output lane (8, 16 or 32)
    //   LANES : number of lanes per 32-bit word (4, 2 or 1)
    // Each lane owns a 2*W-bit slice of the 64-bit accumulator (wide enough for a W x W product).
    // For OUT, the saturated lane value is gathered into one 32-bit word and pushed once at the end.
    template <int W, int LANES>
    void lane_op(bool is_mac, bool is_pass, bool is_rnd, bool is_out,
                 NVUINTW(spec::Instruction::ProcessingElementInstructionConfiguration::VALUE_BITWIDTH) value,
                 spec::VectorType vec_word, spec::VectorType mat_word)
    {
        spec::VectorType out_word = 0;

        #pragma hls_unroll yes
        for (int k = 0; k < LANES; k++) {
            // Sign-extended operands and this lane's accumulator slice.
            NVINTW(W)     a   = nvhls::get_slc<W>(vec_word, k * W);
            NVINTW(W)     b   = nvhls::get_slc<W>(mat_word, k * W);
            NVINTW(2 * W) lane = nvhls::get_slc<2 * W>(accumulation_register, k * 2 * W);

            if (is_mac) {
                // Multiply-accumulate; the sum wraps inside the 2*W-bit lane.
                NVINTW(2 * W) sum = lane + a * b;
                accumulation_register.set_slc(k * 2 * W, sum);
            } else if (is_pass) {
                // Load the vector operand into the lane (matrix operand is ignored).
                NVINTW(2 * W) val = a;
                accumulation_register.set_slc(k * 2 * W, val);
            } else if (is_rnd) {
                // Arithmetic right shift by `value` with round-half-to-even:
                // guard = last bit shifted out, sticky = OR of all lower bits shifted out.
                NVINTW(64) x = lane;
                NVINTW(64) q = x >> value;
                bool guard  = (value != 0) ? (bool)((x >> (value - 1)) & 1) : false;
                NVINTW(64) low_mask = (NVINTW(64)(1) << (value > 0 ? value - 1 : 0)) - 1;
                bool sticky = (value > 1) ? ((x & low_mask) != 0) : false;
                bool round_up = guard && (sticky || (bool)(q & 1));
                NVINTW(64) result = q + (round_up ? 1 : 0);
                NVINTW(2 * W) val = result;
                accumulation_register.set_slc(k * 2 * W, val);
            } else if (is_out) {
                // Clamp the 2*W-bit lane into the signed W-bit range, then pack it.
                NVINTW(64) x = lane;
                NVINTW(64) hi = (NVINTW(64)(1) << (W - 1)) - 1;
                NVINTW(64) lo = -(NVINTW(64)(1) << (W - 1));
                NVINTW(64) clamped = (x > hi) ? hi : ((x < lo) ? lo : x);
                NVINTW(W) val = clamped;
                out_word.set_slc(k * W, val);
            }
        }

        if (is_out) {
            vector_output.Push(out_word);
        }
    }

    void run()
    {
        pe_inst_in.Reset();
        vector_input.Reset();
        matrix_input.Reset();
        vector_output.Reset();

        accumulation_register = 0;

        wait();

        while (1) {
            wait();


            //////////////// START IMPLEMENTATION HERE ////////////////
            // One iteration = one clock cycle. Try to take a new instruction without
            // waiting (PopNB); if none arrived this cycle, the PE simply idles.
            spec::Instruction::ProcessingElementInstruction::Packed_t inst_packed;
            if (pe_inst_in.PopNB(inst_packed)) {
                spec::Instruction::ProcessingElementInstruction inst(inst_packed);

                // The instruction carries a mode (data width of each lane), an opcode and a
                // 5-bit value. For opcode NO_VALUE the value selects MAC/NOP/OUT/PASS/CLR;
                // for opcode RND the value is the right-shift amount.
                const bool is_rnd = (inst.opcode() == spec::Instruction::PEI::RND);
                const bool is_op0 = (inst.opcode() == spec::Instruction::PEI::NO_VALUE);
                const bool is_mac  = is_op0 && (inst.value() == spec::Instruction::PEI::MAC);
                const bool is_pass = is_op0 && (inst.value() == spec::Instruction::PEI::PASS);
                const bool is_out  = is_op0 && (inst.value() == spec::Instruction::PEI::OUT);
                const bool is_clr  = is_op0 && (inst.value() == spec::Instruction::PEI::CLR);
                // (NOP matches none of these flags and therefore changes nothing.)

                // Operands are only sent by the MainBuffer for MAC and PASS, so these are the
                // only instructions allowed to do a blocking Pop. Each operand word packs
                // 4 (INT8), 2 (INT16) or 1 (INT32) lanes.
                spec::VectorType vec_word = 0;
                spec::VectorType mat_word = 0;
                if (is_mac || is_pass) {
                    vec_word = vector_input.Pop();
                    mat_word = matrix_input.Pop();
                }

                if (is_clr) {
                    accumulation_register = 0;
                } else {
                    // The lane width is a compile-time constant inside lane_op, so dispatch on mode.
                    if (inst.mode() == spec::Instruction::Mode::opcode(spec::Instruction::Mode::INT8)) {
                        lane_op<8, 4>(is_mac, is_pass, is_rnd, is_out, inst.value(), vec_word, mat_word);
                    } else if (inst.mode() == spec::Instruction::Mode::opcode(spec::Instruction::Mode::INT16)) {
                        lane_op<16, 2>(is_mac, is_pass, is_rnd, is_out, inst.value(), vec_word, mat_word);
                    } else if (inst.mode() == spec::Instruction::Mode::opcode(spec::Instruction::Mode::INT32)) {
                        lane_op<32, 1>(is_mac, is_pass, is_rnd, is_out, inst.value(), vec_word, mat_word);
                    }
                }
            }

            //////////////// END IMPLEMENTATION HERE ////////////////
        }
    }
};
#endif // ___PROCESSING_ELEMENT_H__