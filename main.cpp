#include "fstream"
#include "string"
#include "vector"
#include "tuple"

int main() {
    std::ifstream infile("T-shape.obj");
    std::ofstream outfile("T-shape-modified-merged.obj");
    std::string v;
    std::vector<std::tuple<std::string, std::string, std::string>> vertices;
    int index = 1;
    while (infile >> v) {
        if (v == "v") {
            std::string x, y, z;
            infile >> x >> y >> z;
            vertices.push_back({x, y, z});
            // outfile << "v" << " " << x << " " << y << " " << z << std::endl;
        } else {
            int x, y, z;
            infile >> x >> y >> z;
            outfile << "v" << " " << std::get<0>(vertices[x]) << " " << std::get<1>(vertices[x]) << " " << std::get<2>(vertices[x]) << std::endl;
            outfile << "v" << " " << std::get<0>(vertices[y]) << " " << std::get<1>(vertices[y]) << " " << std::get<2>(vertices[y]) << std::endl;
            outfile << "v" << " " << std::get<0>(vertices[z]) << " " << std::get<1>(vertices[z]) << " " << std::get<2>(vertices[z]) << std::endl;
            outfile << "f " << index << " " << index + 1 << " " << index + 2 << std::endl;
            // outfile << "f " << x + 1 << " " << y + 1 << " " << z + 1 << std::endl;
            index += 3;
        }
    }
    return 0;
}