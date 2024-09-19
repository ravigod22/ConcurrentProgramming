#include <bits/stdc++.h>
using namespace std;

const int N = 2000000;
mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());

double doubleRand(double limit) {
    uniform_real_distribution<double> rand(1.0, limit);
    return rand(rng);
}

int integerRand(int limit) {
    uniform_int_distribution<int> rand(0, limit);
    return rand(rng);
}


int main() {
    cout << fixed << setprecision(5);
    for (int i = 0; i < N; ++i) {
        cout << doubleRand(3) << " " << doubleRand(3) << " " << (char)(integerRand(1) + 'A') << "\n";
    }
    return 0;
}
