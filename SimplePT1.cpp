#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <random>
#include <vector>
#include <omp.h>
#define PI 3.1415926535897932384626433832795

//Hoang Nguyen - 86865361
//Jeremy Parnell - 

/*
 * Thread-safe random number generator
 */

struct RNG {
    RNG() : distrb(0.0, 1.0), engines() {}

    void init(int nworkers) {
        engines.resize(nworkers);
#if 0
        std::random_device rd;
        for ( int i = 0; i < nworkers; ++i )
            engines[i].seed(rd());
#else
        std::seed_seq seq{ 1234 };
        std::vector<std::uint32_t> seeds(nworkers);
        seq.generate(seeds.begin(), seeds.end());
        for ( int i = 0; i < nworkers; ++i ) engines[i].seed(seeds[i]);
#endif
    }

    double operator()() {
        int id = omp_get_thread_num();
        return distrb(engines[id]);
    }

    std::uniform_real_distribution<double> distrb;
    std::vector<std::mt19937> engines;
} rng;


/*
 * Basic data types
 */

struct Vec {
    double x, y, z;

    Vec(double x_ = 0, double y_ = 0, double z_ = 0) { x = x_; y = y_; z = z_; }

    Vec operator+ (const Vec &b) const  { return Vec(x+b.x, y+b.y, z+b.z); }
    Vec operator- (const Vec &b) const  { return Vec(x-b.x, y-b.y, z-b.z); }
    Vec operator* (double b) const      { return Vec(x*b, y*b, z*b); }

    Vec mult(const Vec &b) const        { return Vec(x*b.x, y*b.y, z*b.z); }
    Vec& normalize()                    { return *this = *this * (1.0/std::sqrt(x*x+y*y+z*z)); }
    double dot(const Vec &b) const      { return x*b.x+y*b.y+z*b.z; }
    Vec cross(const Vec&b) const        { return Vec(y*b.z-z*b.y, z*b.x-x*b.z, x*b.y-y*b.x); }
};

struct Ray {
    Vec o, d;
    Ray(Vec o_, Vec d_) : o(o_), d(d_) {}
};

struct BRDF {
    virtual Vec eval(const Vec &n, const Vec &o, const Vec &i) const = 0;
    virtual void sample(const Vec &n, const Vec &o, Vec &i, double &pdf) const = 0;
};


/*
 * Utility functions
 */

inline double clamp(double x)   {
    return x < 0 ? 0 : x > 1 ? 1 : x;
}

inline int toInt(double x) {
    return static_cast<int>(std::pow(clamp(x), 1.0/2.2)*255+.5);
}


/*
 * Shapes
 */

struct Sphere {
    Vec p, e;           // position, emitted radiance
    double rad;         // radius
    const BRDF &brdf;   // BRDF

    Sphere(double rad_, Vec p_, Vec e_, const BRDF &brdf_) :
        rad(rad_), p(p_), e(e_), brdf(brdf_) {}

    double intersect(const Ray &r) const { // returns distance, 0 if nohit
        Vec op = p-r.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
        double t, eps = 1e-4, b = op.dot(r.d), det = b*b-op.dot(op)+rad*rad;
        if ( det<0 ) return 0; else det = sqrt(det);
        return (t = b-det)>eps ? t : ((t = b+det)>eps ? t : 0);
    }
};


/*
 * Sampling functions
 */

inline void createLocalCoord(const Vec &n, Vec &u, Vec &v, Vec &w) {
    w = n;
    u = ((std::abs(w.x)>.1 ? Vec(0, 1) : Vec(1)).cross(w)).normalize();
    v = w.cross(u);
}

/*
 * Uniform Random PSA Function
 */

Vec uniRandomPSA(const Vec &n)
{
	double z, r, phi, x, y;
	Vec u, v, w;
	z = sqrt(rng());
	r = sqrt(1.0 - z * z);
	phi = 2.0 * PI * rng();
	x = r * cos(phi);
	y = r * sin(phi);

	createLocalCoord(n, u, v, w);
	return u*x + v*y + w*z;
}

/*
 * BRDFs
 */

// Ideal diffuse BRDF
struct DiffuseBRDF : public BRDF {
    DiffuseBRDF(Vec kd_) : kd(kd_) {}

    Vec eval(const Vec &n, const Vec &o, const Vec &i) const {
        return kd * (1.0/PI);
    }

    void sample(const Vec &n, const Vec &o, Vec &i, double &pdf) const {
        i = uniRandomPSA(n);
		pdf = i.dot(n) / PI;
    }

    Vec kd;
};

void luminaireSample(const Sphere &obj, Vec &n, Vec &y1, double &pdf){
    double lam1 = rng();
    double lam2 = rng();
    double z = 2.0 * lam1 - 1.0;
    double x = sqrt(1.0 - pow(z,2)) * cos(2.0 * PI * lam2);
    double y = sqrt(1.0 - pow(z,2)) * sin(2.0 * PI * lam2); 
    n = Vec(x,y,z).normalize();
    y1 = n * obj.rad + obj.p;
    pdf = 1 / (4 * PI * pow(obj.rad,2));
}

/*
 * Scene configuration
 */

// Pre-defined BRDFs
const DiffuseBRDF leftWall(Vec(.75,.25,.25)),
                  rightWall(Vec(.25,.25,.75)),
                  otherWall(Vec(.75,.75,.75)),
                  blackSurf(Vec(0.0,0.0,0.0)),
                  brightSurf(Vec(0.9,0.9,0.9));


// Scene: list of spheres
const Sphere spheres[] = {
    Sphere(1e5,  Vec(1e5+1,40.8,81.6),   Vec(),         leftWall),   // Left
    Sphere(1e5,  Vec(-1e5+99,40.8,81.6), Vec(),         rightWall),  // Right
    Sphere(1e5,  Vec(50,40.8, 1e5),      Vec(),         otherWall),  // Back
    Sphere(1e5,  Vec(50, 1e5, 81.6),     Vec(),         otherWall),  // Bottom
    Sphere(1e5,  Vec(50,-1e5+81.6,81.6), Vec(),         otherWall),  // Top
    Sphere(16.5, Vec(27,16.5,47),        Vec(),         brightSurf), // Ball 1
    Sphere(16.5, Vec(73,16.5,78),        Vec(),         brightSurf), // Ball 2
    Sphere(5.0,  Vec(50,70.0,81.6),      Vec(50,50,50), blackSurf)   // Light
};

// Camera position & direction
const Ray cam(Vec(50, 52, 295.6), Vec(0, -0.042612, -1).normalize());


/*
 * Global functions
 */

bool intersect(const Ray &r, double &t, int &id) {
    double n = sizeof(spheres)/sizeof(Sphere), d, inf = t = 1e20;
    for ( int i = int(n); i--;) if ( (d = spheres[i].intersect(r))&&d<t ) { t = d; id = i; }
    return t<inf;
}

Vec reflectedRadiance(const Ray &r, int depth) {
    double t;                                   // Distance to intersection
    int id = 0;                                 // id of intersected sphere

    if ( !intersect(r, t, id) ) return Vec();   // if miss, return black
    const Sphere &obj = spheres[id];            // the hit object

    Vec x = r.o + r.d*t;                        // The intersection point
    Vec o = (Vec() - r.d).normalize();          // The outgoing direction (= -r.d)

    Vec nx = (x - obj.p).normalize();            // The normal direction
    if ( nx.dot(o) < 0 ) nx = nx*-1.0;
    const BRDF &brdf = obj.brdf;                // Surface BRDF at x
	
	//Calculating Direct Radiance
	const Sphere &obj1 = spheres[7];
	Vec y1, ny, w1, neg_w1;
	double pdf1, r2, vis = 0.0;
	luminaireSample(obj1, ny, y1, pdf1);
	w1 = (y1 - x).normalize();
	neg_w1 = Vec(-w1.x, -w1.y, -w1.z);
	r2 = (y1-x).dot(y1-x);

	int id1 = 0, id2 = 0, temp, temp1;
	Ray int_x(x, w1);
	Ray int_y(y1, neg_w1);
	temp = intersect(int_x, t, id1);
	temp1 = intersect(int_y, t, id2);
	if (id1 == 7 and id2 == id and temp == 1 and temp1 == 1)
	{
		vis = 1.0;
	}
	

	Vec Le = obj1.e;
	Vec reflRad = Le.mult(brdf.eval(x, o, w1)) * vis * nx.dot(w1) * ny.dot(neg_w1) * (1.0 / (r2*pdf1));

	//Calculating Indirect Radiance
	int rrDepth = 5;
	double survivalProb = 0.9, p, pdf2;
	Vec w2;

	if (depth <= rrDepth)
	{
		p = 1.0;
	}
	else
	{
		p = survivalProb;
	}

	if (rng() < p)
	{
		brdf.sample(nx, o, w2, pdf2);
		Ray y2(x, w2);
		reflRad = reflRad + (reflectedRadiance(y2, depth+1).mult(brdf.eval(nx, o, w2))) * (nx.dot(w2) * (1.0 / (pdf2*p)));
	}
	return reflRad;

}

/*
 * KEY FUNCTION: radiance estimator
 */

Vec receivedRadiance(const Ray &r, int depth, bool flag) {
    double t;                                   // Distance to intersection
    int id = 0;                                 // id of intersected sphere

    if ( !intersect(r, t, id) ) return Vec();   // if miss, return black
    const Sphere &obj = spheres[id];            // the hit object

    Vec x = r.o + r.d*t;                        // The intersection point
    Vec o = (Vec() - r.d).normalize();          // The outgoing direction (= -r.d)

    Vec n = (x - obj.p).normalize();            // The normal direction
    if ( n.dot(o) < 0 ) n = n*-1.0;

    Vec Le = obj.e;                             // Emitted radiance
    const BRDF &brdf = obj.brdf;                // Surface BRDF at x

    //2. Call brdf.sample() to sample an incoming direction and continue the recursion
	return Le + reflectedRadiance(r, depth);
}


/*
 * Main function (do not modify)
 */

int main(int argc, char *argv[]) {
    int nworkers = omp_get_num_procs();
    omp_set_num_threads(nworkers);
    rng.init(nworkers);

    int w = 480, h = 360, samps = argc==2 ? atoi(argv[1])/4 : 1; // # samples
    Vec cx = Vec(w*.5135/h), cy = (cx.cross(cam.d)).normalize()*.5135;
    std::vector<Vec> c(w*h);

#pragma omp parallel for schedule(dynamic, 1)
    for ( int y = 0; y < h; y++ ) {
        for ( int x = 0; x < w; x++ ) {
            const int i = (h - y - 1)*w + x;

            for ( int sy = 0; sy < 2; ++sy ) {
                for ( int sx = 0; sx < 2; ++sx ) {
                    Vec r;
                    for ( int s = 0; s<samps; s++ ) {
                        double r1 = 2*rng(), dx = r1<1 ? sqrt(r1)-1 : 1-sqrt(2-r1);
                        double r2 = 2*rng(), dy = r2<1 ? sqrt(r2)-1 : 1-sqrt(2-r2);
                        Vec d = cx*(((sx+.5 + dx)/2 + x)/w - .5) +
                            cy*(((sy+.5 + dy)/2 + y)/h - .5) + cam.d;
                        r = r + receivedRadiance(Ray(cam.o, d.normalize()), 1, true)*(1./samps);
                    }
                    c[i] = c[i] + Vec(clamp(r.x), clamp(r.y), clamp(r.z))*.25;
                }
            }
        }
#pragma omp critical
        fprintf(stderr,"\rRendering (%d spp) %6.2f%%",samps*4,100.*y/(h-1));
    }
    fprintf(stderr, "\n");

    // Write resulting image to a PPM file
    FILE *f = fopen("image.ppm", "w");
    fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
    for ( int i = 0; i<w*h; i++ )
        fprintf(f, "%d %d %d ", toInt(c[i].x), toInt(c[i].y), toInt(c[i].z));
    fclose(f);

    return 0;
}
