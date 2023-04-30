#include <gunrock/algorithms/algorithms.hxx>
#include <gunrock/formats/formats.hxx>

using namespace gunrock;
using namespace memory;

int main()
{
    using vertex_t = int;
    using edge_t = int;
    using weight_t = float;

    using csr_t = format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;

    return 0;
}