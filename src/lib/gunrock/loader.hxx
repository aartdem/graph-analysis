#include <filesystem>
#include <fstream>      
#include <sstream>      
#include <stdexcept>    
#include <string>       
#include <vector>

namespace algos {
namespace detail {
// Reads an *undirected* graph in MatrixMarket (.mtx) coordinate format.
// The MTX format is assumed to be 1-based indexing.
// On return, rows[i], cols[i], vals[i] are the COO entries.
template <typename Vt, typename Et, typename Wt>
void load_mtx_coo(const std::filesystem::path &path, std::vector<Vt> &rows,
                  std::vector<Vt> &cols, std::vector<Wt> &vals) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("Cannot open MTX file: " + path.string());

  std::string line;
  // Skip comments
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '%')
      continue;
    // this line should be the header: M N nnz
    std::istringstream iss(line);
    int M, N, nnz;
    if (!(iss >> M >> N >> nnz))
      throw std::runtime_error("Invalid MTX header in " + path.string());
    rows.reserve(nnz);
    cols.reserve(nnz);
    vals.reserve(nnz);
    // Read the nnz entries
    for (int i = 0; i < nnz; ++i) {
      int u, v;
      Wt w;
      in >> u >> v >> w;
      // convert 1-based to 0-based
      rows.push_back(static_cast<Vt>(u - 1));
      cols.push_back(static_cast<Vt>(v - 1));
      vals.push_back(w);
    }
    return;
  }
  throw std::runtime_error("Empty or malformed MTX: " + path.string());
}
} // namespace detail
} // namespace algos