#include "fstream"
#include "string"
#include "vector"
#include "tuple"
#include "iomanip"

int main() {
    std::ifstream infile("T-shape.obj");
    std::ofstream outfile("T-shape-modified.obj");
    std::string v;
    std::vector<std::tuple<double, double, double>> vertices;
    int index = 1;
    while (infile >> v) {
        if (v == "v") {
            double x, y, z;
            infile >> x >> y >> z;
            vertices.push_back({x, y, z});
            // outfile << "v" << " " << x << " " << y << " " << z << std::endl;
        } else {
            int x, y, z;
            infile >> x >> y >> z;
            outfile << std::fixed << std::setprecision(2);
            outfile << "v" << " " << std::get<0>(vertices[x]) << " " << std::get<1>(vertices[x]) - 0.04 << " " << std::get<2>(vertices[x]) << std::endl;
            outfile << "v" << " " << std::get<0>(vertices[y]) << " " << std::get<1>(vertices[y]) - 0.04 << " " << std::get<2>(vertices[y]) << std::endl;
            outfile << "v" << " " << std::get<0>(vertices[z]) << " " << std::get<1>(vertices[z]) - 0.04 << " " << std::get<2>(vertices[z]) << std::endl;
            outfile << "f " << index << " " << index + 1 << " " << index + 2 << std::endl;
            // outfile << "f " << x + 1 << " " << y + 1 << " " << z + 1 << std::endl;
            index += 3;
        }
    }
    return 0;
}