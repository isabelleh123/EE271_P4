#ifndef __MAIN_BUFFER_H__
#define __MAIN_BUFFER_H__

#pragma once

#include <systemc.h>
#include <nvhls_connections.h>
#include <vector>

#include "../Spec.h"
#include "../InstructionSpec.h"

SC_MODULE(MainBuffer)
{
public:
        
    sc_in<bool> clk;
    sc_in<bool> rst;

    Connections::In<spec::Instruction::MemoryInstruction::Packed_t> buf_inst_in;
    Connections::In<spec::VectorType> output_data_in[spec::N_PE];

    Connections::Out<spec::VectorType> matrix_data_out[spec::N_PE];
    Connections::Out<spec::VectorType> vector_data_out[spec::N_PE];

    Connections::In<spec::VectorType> matrix_mem_write;
    Connections::In<NVUINTW(spec::kMemAddrWidth)> matrix_mem_write_addr;

    Connections::In<spec::VectorType> vector_mem_write;
    Connections::In<NVUINTW(spec::kMemAddrWidth)> vector_mem_write_addr;

    Connections::Out<spec::VectorType> output_mem_read;
    Connections::In<NVUINTW(spec::kMemAddrWidth)> output_mem_read_addr;

    spec::VectorType matrix_mem[spec::kMemDepth];
    spec::VectorType vector_mem[spec::kMemDepth];
    spec::VectorType output_mem[spec::kMemDepth];

    NVUINTW(spec::kMemAddrWidth) addr = 0;
        spec::VectorType data = 0;
    


    SC_HAS_PROCESS(MainBuffer);
    MainBuffer(sc_module_name name)
        : sc_module(name),
          clk("clk"),
          rst("rst")
    {
        for (unsigned i = 0; i < spec::kMemDepth; ++i) {
            matrix_mem[i] = 0;
            vector_mem[i] = 0;
            output_mem[i] = 0;
        }

        SC_THREAD(run);
        sensitive << clk.pos();
        async_reset_signal_is(rst, false);

    }

    void run() {
        buf_inst_in.Reset();
        matrix_mem_write.Reset();
        matrix_mem_write_addr.Reset();
        vector_mem_write.Reset();
        vector_mem_write_addr.Reset();
        output_mem_read_addr.Reset();
        output_mem_read.Reset();
        
        #pragma hls_unroll yes
        for (int i=0; i < spec::N_PE; i++) {
            output_data_in[i].Reset();
            matrix_data_out[i].Reset();
            vector_data_out[i].Reset(); // Reset all vector_data_out ports
        }
        
        wait();


        while(1) {
            wait();
            
            if (matrix_mem_write_addr.PopNB(addr)) {
                if (matrix_mem_write.PopNB(data)) {
                    matrix_mem[addr] = data;
                }
            }

            if (vector_mem_write_addr.PopNB(addr)) {
                if (vector_mem_write.PopNB(data)) {
                    vector_mem[addr] = data;
                }
            }

            if (output_mem_read_addr.PopNB(addr)) {
                output_mem_read.Push(output_mem[addr]);
            }

            
            spec::Instruction::MemoryInstruction::Packed_t inst_packed;
            if (buf_inst_in.PopNB(inst_packed)) {
                spec::Instruction::MemoryInstruction buf_inst(inst_packed);

                //////////////// START IMPLEMENTATION HERE ////////////////
                // Matrix and output memories are organised in rows of N_PE consecutive words:
                // row `mema_offset` starts at word mema_offset * N_PE, and PE i uses word + i.
                const NVUINTW(spec::kMemAddrWidth) row_base = buf_inst.mema_offset() * spec::N_PE;

                if (buf_inst.opcode() == spec::Instruction::MI::READ) {
                    // Vector memory words hold 1 (INT32), 2 (INT16) or 4 (INT8) packed elements.
                    // memb_offset counts elements, so its low bits select the element inside a word
                    // and its upper bits select the word.
                    spec::VectorType vec_word;
                    if (buf_inst.mode() == spec::Instruction::Mode::opcode(spec::Instruction::Mode::INT8)) {
                        spec::VectorType w = vector_mem[buf_inst.memb_offset() >> 2];
                        NVUINTW(8) elem = nvhls::get_slc<8>(w, (buf_inst.memb_offset() & 3) * 8);
                        vec_word = (elem, elem, elem, elem);      // repeat the byte in all 4 lanes
                    } else if (buf_inst.mode() == spec::Instruction::Mode::opcode(spec::Instruction::Mode::INT16)) {
                        spec::VectorType w = vector_mem[buf_inst.memb_offset() >> 1];
                        NVUINTW(16) elem = nvhls::get_slc<16>(w, (buf_inst.memb_offset() & 1) * 16);
                        vec_word = (elem, elem);                  // repeat the half-word in both lanes
                    } else {
                        vec_word = vector_mem[buf_inst.memb_offset()];
                    }

                    // Send one matrix word to each PE and broadcast the same vector word to all PEs.
                    #pragma hls_unroll yes
                    for (int i = 0; i < spec::N_PE; i++) {
                        matrix_data_out[i].Push(matrix_mem[row_base + i]);
                        vector_data_out[i].Push(vec_word);
                    }
                } else if (buf_inst.opcode() == spec::Instruction::MI::WRITE) {
                    // Collect one result from every PE (blocks until its OUT instruction pushes it)
                    // and store the results as the output row.
                    #pragma hls_unroll yes
                    for (int i = 0; i < spec::N_PE; i++) {
                        output_mem[row_base + i] = output_data_in[i].Pop();
                    }
                }
                // NOP: nothing to do.

                //////////////// END IMPLEMENTATION HERE ////////////////
            }
        }
    }
};


#endif // __MAIN_BUFFER_H__