#ifndef HTTPSTAN_SOCKET_WRITER_HPP
#define HTTPSTAN_SOCKET_WRITER_HPP

#include "callbacks_writer.pb.h"
#include <boost/asio.hpp>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <iostream>
#include <stan/callbacks/writer.hpp>
#include <string>
#include <vector>

/**
 * NOTE: httpstan makes use of `message_prefix` in an unexpected way!
 *
 * httpstan uses `message_prefix` to record what messages the socket_writer instance is receiving.
 * In a call to `hmc_nuts_diag_e_adapt`, three socket_writers are used:
 * 1. init_writer
 * 2. sample_writer
 * 3. diagnostic_writer
 *
 * httpstan uses `message_prefix` to allow the socket_writer to know in what
 * context it is being used.  identity of the socket_writer. For example, the
 * diagnostic writer uses the string `diagnostic_writer:` (note the colon) as
 * its message_prefix.
 *
 * Additional background:
 *
 * Much of the code here is involved in parsing the output of the callback
 * writers used by stan::services functions.  For example,
 * stan::services::sample::hmc_nuts_diag_e_adapt writes messages to the
 * following five writers:
 * - ``init_writer`` Writer callback for unconstrained inits
 * - ``sample_writer`` Writer for draws
 * - ``diagnostic_writer`` Writer for diagnostic information
 *
 * ``sample_writer`` and ``diagnostic_writer`` receive messages in a predictable fashion: headers followed by samples.
 * For example:
 *   sample_writer:["lp__","accept_stat__","stepsize__","treedepth__","n_leapfrog__","divergent__","energy__","y"]
 *   sample_writer:[-3.16745e-06,0.999965,1,2,3,0,0.0142087,0.00251692]
 * If adaptation happens, however, ``sample_writer`` receives messages similar to
 * the following after the header but before the draws:
 *   sample_writer:"Adaptation terminated"
 *   sample_writer:"Step size = 0.809818"
 *   sample_writer:"Diagonal elements of inverse mass matrix:"
 *   sample_writer:0.961989
 *
 */

namespace stan {
namespace callbacks {

/* Enum used by sample writer only. Keeps track of state. */
enum class ProcessingAdaptationState {
  BEFORE_PROCESSING_ADAPTATION, // if no adaptation, stay here
  PROCESSING_ADAPTATION,
  FINAL_ADAPTATION_MESSAGE,
  AFTER_PROCESSING_ADAPTATION
};

/**
 * <code>socket_writer</code> is an implementation
 * of <code>writer</code> that writes Protobuf-encoded values to a socket.
 */
class socket_writer : public writer {
private:
  /**
   * Output
   */

  boost::asio::io_service io_service;
  boost::asio::local::stream_protocol::socket socket;

  /**
   * Channel name with which to prefix strings sent to the socket.
   */
  std::string message_prefix_;
  std::vector<std::string> diagnostic_fields_;
  std::vector<std::string> sample_fields_;
  ProcessingAdaptationState processing_adaptation_state_ = ProcessingAdaptationState::BEFORE_PROCESSING_ADAPTATION;

  /**
   * Send a protocol buffer message to a socket using length-prefix encoding.
   */
  size_t send_message(const stan::WriterMessage &message, boost::asio::local::stream_protocol::socket &socket) {
    boost::asio::streambuf stream_buffer;
    std::ostream output_stream(&stream_buffer);
    {
      ::google::protobuf::io::OstreamOutputStream raw_output_stream(&output_stream);
      ::google::protobuf::io::CodedOutputStream coded_output_stream(&raw_output_stream);
      coded_output_stream.WriteVarint32(message.ByteSizeLong());
      message.SerializeToCodedStream(&coded_output_stream);
      // IMPORTANT: In order to flush a CodedOutputStream it must be deleted.
    }
    return socket.send(stream_buffer.data());
  }

public:
  /**
   * Constructs a writer with an output socket
   * and an optional prefix for comments.
   *
   * @param[in, out] output ostream
   * @param[in] message_prefix will be prefixed to each string which is sent to the socket. Default is "".
   */
  explicit socket_writer(const std::string &socket_filename, const std::string &message_prefix = "")
      : socket(io_service), message_prefix_(message_prefix) {
    boost::asio::local::stream_protocol::endpoint ep(socket_filename);
    socket.connect(ep);
  }

  /**
   * Destructor
   */
  ~socket_writer() { socket.close(); }

  /**
   * Writes a sequence of names.
   *
   * @param[in] names Names in a std::vector
   */
  void operator()(const std::vector<std::string> &names) {
    std::vector<std::string>::const_iterator last = names.end();
    if (message_prefix_ == "diagnostic_writer:") {
      if (diagnostic_fields_.empty()) {
        for (std::vector<std::string>::const_iterator it = names.begin(); it != last; ++it) {
          diagnostic_fields_.push_back(*it);
        }
        return;
      }
      stan::WriterMessage writer_message;
      writer_message.set_topic(stan::WriterMessage_Topic_DIAGNOSTIC);

      stan::WriterMessage_Feature *feature = writer_message.add_feature();
      stan::WriterMessage_BytesList *bytes_list = new stan::WriterMessage_BytesList;
      for (std::vector<std::string>::const_iterator it = names.begin(); it != last; ++it) {
        bytes_list->add_value(*it);
      }
      feature->set_allocated_bytes_list(bytes_list);

      send_message(writer_message, socket);
      return;
    } else if (message_prefix_ == "init_writer:") {
      throw std::runtime_error("Unexpected string vector for init writer.");
    } else if (message_prefix_ == "sample_writer:") {
      // sample writer receives only one string vector message, the column header
      if (!sample_fields_.empty())
        throw std::runtime_error("Unexpected string vector in sample writer after column header.");
      for (std::vector<std::string>::const_iterator it = names.begin(); it != last; ++it) {
        sample_fields_.push_back(*it);
      }
      return;
    }
  }

  /**
   * Writes a set of values.
   *
   * @param[in] state Values in a std::vector
   */
  void operator()(const std::vector<double> &state) {
    std::vector<double>::const_iterator last = state.end();

    if (message_prefix_ == "diagnostic_writer:") {
      if (diagnostic_fields_.empty()) {
        throw std::runtime_error("diagnostic fields must be set before receiving values");
      }

      stan::WriterMessage writer_message;
      writer_message.set_topic(stan::WriterMessage_Topic_DIAGNOSTIC);
      for (std::size_t i = 0; i < diagnostic_fields_.size(); ++i) {
        stan::WriterMessage_Feature *feature = writer_message.add_feature();
        feature->set_name(diagnostic_fields_[i]);
        stan::WriterMessage_DoubleList *double_list = new stan::WriterMessage_DoubleList;
        double_list->add_value(state[i]);
        feature->set_allocated_double_list(double_list);
      }

      send_message(writer_message, socket);
      return;
    } else if (message_prefix_ == "init_writer:") {
      stan::WriterMessage writer_message;
      writer_message.set_topic(stan::WriterMessage_Topic_INITIALIZATION);

      stan::WriterMessage_Feature *feature = writer_message.add_feature();
      stan::WriterMessage_DoubleList *double_list = new stan::WriterMessage_DoubleList;
      for (std::vector<double>::const_iterator it = state.begin(); it != last; ++it) {
        double_list->add_value(*it);
      }
      feature->set_allocated_double_list(double_list);

      send_message(writer_message, socket);
      return;
    } else if (message_prefix_ == "sample_writer:") {
      if (sample_fields_.empty())
        throw std::runtime_error("Sample fields should be populated before sample writer writes a vector of doubles.");

      if ((processing_adaptation_state_ == ProcessingAdaptationState::PROCESSING_ADAPTATION) ||
          (processing_adaptation_state_ == ProcessingAdaptationState::FINAL_ADAPTATION_MESSAGE))
        throw std::runtime_error("Adaptation should have completed before sample writer writes a vector of doubles.");

      stan::WriterMessage writer_message;
      writer_message.set_topic(stan::WriterMessage_Topic_SAMPLE);

      for (std::size_t i = 0; i < sample_fields_.size(); ++i) {
        stan::WriterMessage_Feature *feature = writer_message.add_feature();
        feature->set_name(sample_fields_[i]);
        stan::WriterMessage_DoubleList *double_list = new stan::WriterMessage_DoubleList;
        double_list->add_value(state[i]);
        feature->set_allocated_double_list(double_list);
      }

      send_message(writer_message, socket);
      return;
    }
  }

  /**
   * Writes the message_prefix to the stream followed by a newline.
   */
  void operator()() {
    // unused
    return;
  }

  /**
   * Writes the message_prefix then the message.
   *
   * @param[in] message A string
   */
  void operator()(const std::string &message) {
    if (message_prefix_ == "diagnostic_writer:") {
      // DEBUG: prototype here
      stan::WriterMessage writer_message;
      writer_message.set_topic(stan::WriterMessage_Topic_DIAGNOSTIC);

      stan::WriterMessage_Feature *feature = writer_message.add_feature();
      stan::WriterMessage_BytesList *bytes_list = new stan::WriterMessage_BytesList;
      bytes_list->add_value(message);
      feature->set_allocated_bytes_list(bytes_list);

      send_message(writer_message, socket);
      return;
    } else if (message_prefix_ == "init_writer:") {
      throw std::runtime_error("Unexpected string vector for init writer.");
    } else if (message_prefix_ == "sample_writer:") {
      // state machine dance here
      if (processing_adaptation_state_ == ProcessingAdaptationState::BEFORE_PROCESSING_ADAPTATION) {
        if (message.rfind("Adaptation terminated", 0) == 0) {
          // message starts with "Adaptation terminated"
          processing_adaptation_state_ = ProcessingAdaptationState::PROCESSING_ADAPTATION;
        }
      } else if (processing_adaptation_state_ == ProcessingAdaptationState::PROCESSING_ADAPTATION) {
        if (message.rfind("Diagonal elements of inverse mass matrix", 0) == 0) {
          // message starts with "Diagonal elements of inverse mass matrix"
          // the next "message" (vector of doubles) will be the final adaptation message
          processing_adaptation_state_ = ProcessingAdaptationState::FINAL_ADAPTATION_MESSAGE;
        }
      } else if (processing_adaptation_state_ == ProcessingAdaptationState::FINAL_ADAPTATION_MESSAGE) {
        // this message is the last adaptation-related message before normal draws start arriving
        processing_adaptation_state_ = ProcessingAdaptationState::AFTER_PROCESSING_ADAPTATION;
      }

      stan::WriterMessage writer_message;
      writer_message.set_topic(stan::WriterMessage_Topic_SAMPLE);

      stan::WriterMessage_Feature *feature = writer_message.add_feature();
      stan::WriterMessage_BytesList *bytes_list = new stan::WriterMessage_BytesList;
      bytes_list->add_value(message);
      feature->set_allocated_bytes_list(bytes_list);

      send_message(writer_message, socket);
      return;
    }
  }
};

} // namespace callbacks
} // namespace stan
#endif // HTTPSTAN_SOCKET_WRITER_HPP
