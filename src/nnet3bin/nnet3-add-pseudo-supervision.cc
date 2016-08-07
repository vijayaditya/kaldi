// nnet3bin/nnet3-add-recurrent-io-to-egs.cc

// Copyright 2012-2015  Johns Hopkins University (author:  Daniel Povey)
//                2015  Vijay Peddinti

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet3/nnet-example.h"
#include "nnet3/nnet-example-utils.h"
#include "nnet3/nnet-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Modifies the examples to add pseudo supervision for specified nodes.\n"
        "This would be used to compute cost functions which do not require\n"
        "supervision."
        "\n"
        "Usage:  nnet3-add-pseudo-supervision [options] <raw-model-in>"
        " <egs-in> <egs-out>\n"
        "\n"
        "An example:\n"
        "nnet3-add-pseudo-supervision 'L1output,L2output' ark:1.egs ark:- \n";

    bool compress = false;

    ParseOptions po(usage);
    po.Register("compress", &compress, "If true, write egs in "
                "compressed format.");

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }


    std::string node_names_string = po.GetArg(1),
         nnet_rxfilename = po.GetArg(2),
         examples_rspecifier = po.GetArg(3),
         examples_wspecifier = po.GetArg(4);

    std::vector<std::string> split_string;
    SplitStringToVector(node_names_string, ",", true,
                        &split_string);

    Nnet nnet;
    ReadKaldiObject(nnet_rxfilename, &nnet);


    // get the output dim for the nodes specified
    std::vector<int32> output_dims(split_string.size());
    for (size_t i = 0; i < split_string.size(); i++)  {
      KALDI_ASSERT(nnet.IsOutputNode(nnet.GetNodeIndex(split_string[i])));
      int32 output_dim = nnet.OutputDim(split_string[i]);
      if  (output_dim == -1)  {
        std::cerr << "Model does not have node named "
                  << split_string[i] << '\n';
        exit(1);
      }
      output_dims[i] = output_dim;
    }

    SequentialNnetExampleReader example_reader(examples_rspecifier);
    NnetExampleWriter example_writer(examples_wspecifier);
    int64 num_read = 0, num_written = 0;

    while (!example_reader.Done()) {
      Timer tim;//debug
      const std::string &cur_key = example_reader.Key();
      NnetExample cur_eg(example_reader.Value());
      example_reader.Next();
      num_read++;


      // copy the indexes from the NnetIO named "output".
      int32 t_begin = 0, t_size = 0;
      for (std::vector<NnetIo>::const_iterator it = cur_eg.io.begin();
           it != cur_eg.io.end(); it++)  {
        if (it->name == "output") {
          t_begin = it->indexes[0].t;
          t_size = it->indexes.size();
        }
      }

      for (size_t i = 0; i < split_string.size(); i++)  {
        SparseMatrix<BaseFloat> pseudo_output_features(t_size, output_dims[i]);
        NnetIo pseudo_output(split_string[i],
                             t_begin, pseudo_output_features);
        cur_eg.io.push_back(pseudo_output);
      }

      if (compress)
        cur_eg.Compress();
      example_writer.Write(cur_key, cur_eg);
      num_written++;
    }
    KALDI_LOG << "Processed " << num_read << " egs to " << num_written << ".";
    return (num_written != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
