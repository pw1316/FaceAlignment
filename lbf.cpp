#include "lbf.h"
#include "forest.h"
#include "timer.h"
#include "3rdparty/liblinear/linear.h"
#include <omp.h>
#include <string>

using namespace std;

#if DATA_SET == SET_BIOID
const string TRAIN_DATA_PATH = ".\\TrainData\\BioID\\";
const string TEST_DATA_PATH = ".\\TestData\\BioID\\";
inline string getImagePath(int i)
{
    char fname[256];
    sprintf(fname, "BioID_%04d.pgm", i);
    return TRAIN_DATA_PATH + fname;
}
inline string getShapePath(int i)
{
    char fname[256];
    sprintf(fname, "bioid_%04d.pts", i);
    return TRAIN_DATA_PATH + fname;
}
#elif DATA_SET == SET_FACE_WAREHOUSE
const string TRAIN_DATA_PATH = ".\\TrainData\\FaceWarehouse\\";
const string TEST_DATA_PATH = ".\\TestData\\FaceWarehouse\\";
inline string getImagePath(int i)
{
    char fname[256];
    sprintf(fname, "%04d.png", i);
    return TRAIN_DATA_PATH + fname;
}
inline string getShapePath(int i)
{
    char fname[256];
    sprintf(fname, "%04d.land", i);
    return TRAIN_DATA_PATH + fname;
}
#else
#error DATA_SET not specified
#endif

static void mean(const Matrix2Df& data, float* res)
{
    float inv = 1.f / (float)data.cols;

    for (int i = 0; i < data.rows; ++i)
    {
        float sum = 0.f;
        float* dt = data[i];
        for (int j = 0; j < data.cols; ++j)
            sum += dt[j];
        res[i] = sum*inv;
    }
}

class LBF
{
public:
    void train();
    void loadRegressor(Matrix2D<LbfRandomForest>& rforests, Matrix2D<model*>& models);
    void test();
private:
    void loadShape(const string& fpath, int width, int height, float* shp);
    void dumpShape(string& fpath, float* shp);
    void loadTrainingData(Image* Train_Images, Matrix2Df& Train_Shapes, Matrix2Df& Regression_Shapes)
    {
        string img_path, shp_path;

        float _shp[2 * NUM_LANDMARKS];
        for (int i = 0; i < TRAIN_DATA_SIZE; ++i)
        {
            img_path = getImagePath(i);
            shp_path = getShapePath(i);

            Train_Images[i].load(img_path);
            loadShape(shp_path, Train_Images[i].width, Train_Images[i].height, _shp);
            Train_Shapes.column(i) = _shp;
        }
        mean(Train_Shapes, Mean_Shape); // same rectangle??
        string mean_shape_f = TRAIN_DATA_PATH + "meanshape";
        dumpShape(mean_shape_f, Mean_Shape);

        for (int i = 0, iLen = TRAIN_DATA_SIZE*NUM_AUG; i < iLen; ++i)
        {
            int idx = randM(TRAIN_DATA_SIZE);
            Regression_Shapes.column(i) = Train_Shapes.column(idx);
        }
    }
    feature_node* globalFeatures(int* localBF/*,int P*/, int non0count)
    {
        feature_node* feas = new feature_node[non0count + 1];
        for (int j = 0; j < non0count; ++j)
        {
            feas[j].index = localBF[j] + 1; // !! liblinear index starts from 1
            feas[j].value = 1.0;
        }
        feas[non0count].index = -1; // mark end
        return feas;
    }
    static void noprint(const char*) {}
    void ridgeRegression(string fpath, int N, int P, float* _y, const Matrix2Di& _X, float lambda, float* reg_result)
    {
        set_print_string_function(&LBF::noprint); // quiet please
        problem prob;
        prob.l = N; prob.n = P;
        double* y = new double[N];
        for (int i = 0; i < N; ++i) y[i] = _y[i];
        prob.y = y;
        prob.bias = -1.0;
        prob.x = new feature_node*[N];
        int nonzero_leaves = NUM_LANDMARKS*NUM_TREES;
        for (int i = 0; i < N; ++i)
        {
            prob.x[i] = globalFeatures(_X[i]/*,P*/, nonzero_leaves);
        }
        model* md = 0;
#ifdef NO_READ_W
        if (1)
#else
        if (!(md = load_model(fpath.c_str())))
#endif
        {
            parameter params;
            params.solver_type = L2R_L2LOSS_SVR;//_DUAL;
            params.C = 0.5 / (double)lambda;
            params.nr_weight = 0;
            params.p = 0.; // => ridge regression
            params.eps = 0.001; // 0.1 for dual?

            const char* error_msg = check_parameter(&prob, &params);
            if (error_msg)
            {
                printf("%s\n", error_msg);
                exit(-1);
            }
            else
            {
                //tic();
                md = ::train(&prob, &params);
                //toc();
                save_model(fpath.c_str(), md);
            }
        }
        else
        {
            //printf("loaded\n");
        }

        if (md) // 
        {
            for (int i = 0; i < N; ++i)
                reg_result[i] = (float)predict(md, prob.x[i]);
            free_and_destroy_model(&md);
        }



        // clean up
        delete[] prob.y;
        for (int i = 0; i < N; ++i)
            delete[] prob.x[i];
        delete[] prob.x;

    }

    float Mean_Shape[NUM_LANDMARKS * 2];
};


int main()
{
    LBF lbf;
    tic();
    lbf.train();
    toc();
    //lbf.test();
    
    return 0;
}

void LBF::train()
{
    char cmd_buf[256];
#ifdef NO_READ_RF
    sprintf(cmd_buf, "del \"%s*.rf\"", TRAIN_DATA_PATH.c_str());
    printf("exec cmd...:\n\t%s\n", cmd_buf);
    system(cmd_buf);
#endif
#ifdef NO_READ_W
    sprintf(cmd_buf, "del \"%s*.w\"", TRAIN_DATA_PATH.c_str());
    printf("exec cmd...:\n\t%s\n", cmd_buf);
    system(cmd_buf);
#endif

    Image trainImages[TRAIN_DATA_SIZE];
    Matrix2Df trainShapes(NUM_LANDMARKS * 2, TRAIN_DATA_SIZE);
    Matrix2Df regShapes(NUM_LANDMARKS * 2, TRAIN_DATA_SIZE * NUM_AUG);
    Matrix2Df gtOffsets(NUM_LANDMARKS * 2, TRAIN_DATA_SIZE * NUM_AUG);

    loadTrainingData(trainImages, trainShapes, regShapes);

    int Nsamples = regShapes.cols;
    LbfRandomForest rforests[NUM_LANDMARKS];
    int nonzero_leaves[NUM_TREES*NUM_LANDMARKS];

#ifdef DUMP_TRAINING_SHAPES
    sprintf(cmd_buf, "del \"%strain_smpl*.png\"", TRAIN_DATA_PATH.c_str());
    printf("exec cmd...:\n\t%s\n", cmd_buf);
    system(cmd_buf);

    printf("dumping training intermediate states: stage 0, 99\n");
    Matrix2Df shp_i(NUM_LANDMARKS * 2, 1);
    char tshpname[256];
    for (int i = 0; i < Nsamples; ++i)
    {
        sprintf(tshpname, "train_smpl%04d_%02d_t00.png", i / NUM_AUG, i % NUM_AUG);
        shp_i.column(0) = Regression_Shapes.column(i);
        trainImages[i / NUM_AUG].plotShape(shp_i[0]);
        trainImages[i / NUM_AUG].saveAs(TRAIN_DATA_PATH + tshpname);
        // ref
        if (i % NUM_AUG == 0)
        {
            sprintf(tshpname, "train_smpl%04d_%02d_t99.png", i / NUM_AUG, i%NUM_AUG);
            shp_i.column(0) = Train_Shapes.column(i / NUM_AUG);
            trainImages[i / NUM_AUG].plotShape(shp_i[0]);
            trainImages[i / NUM_AUG].saveAs(TRAIN_DATA_PATH + tshpname);
        }
    }
#endif

    for (int t = 1; t <= NUM_STAGES; ++t)
    {
        printf("\n\n==================== Training stage %d ====================\n", t);
        tic();
        // compute ground-truth offset for this stage
        for (int coord_i = 0; coord_i < 2 * NUM_LANDMARKS; ++coord_i)
        {
            float* gt_offset = gtOffsets[coord_i];
            float *pgt = trainShapes[coord_i],
                *preg = regShapes[coord_i];
            for (int i = 0; i < Nsamples; ++i)
            {
                *gt_offset++ = pgt[i / NUM_AUG] - (*preg++);
            }
        }
        printf("#\n");
        toc("update ground-truth offset");

        tic();
        //built all forests for this stage
        int tot_leaves = 0; // for all landmarks
#pragma omp parallel for reduction(+:tot_leaves)
        for (int l = 0; l < NUM_LANDMARKS; ++l)
        {
            char fname_buf[256];
            sprintf(fname_buf, "t%dl%d.rf", t, l);
            string fpath = TRAIN_DATA_PATH + fname_buf;
#ifdef NO_READ_RF
            if (0)
#else
            if (rforests[l].load(fpath))
#endif
            {
                if (0 == omp_get_thread_num())
                    printf("------ Loaded random forest for landmark %d ------\n", l + 1);
            }
            else
            {
                if (0 == omp_get_thread_num())
                    printf("\n------ Building random forest for landmark %d ------\n\n", l + 1);

                Matrix2Df landmarks(regShapes[2 * l], 2, regShapes.cols);
                Matrix2Df offsets(gtOffsets[2 * l], 2, gtOffsets.cols);

                if (0 == omp_get_thread_num()) tic();
                rforests[l].build(trainImages, landmarks, offsets);
                if (0 == omp_get_thread_num()) toc();
                rforests[l].dump(fpath);
            }
            tot_leaves += rforests[l].nLeaves;
        }
        printf("tot_leaves %d\n", tot_leaves);
        toc("built all forests in this stage");

        tic();
        // global binary features
        Matrix2Di Phi(Nsamples, NUM_TREES*NUM_LANDMARKS); // sparse, NUM_TREES*NUM_LANDMARKS non-zeros
                                                          // gather local binary features
        for (int i = 0; i < Nsamples; ++i)
        {
            printf("\rRun rf get local bin features %d/%d   ", i + 1, Nsamples);
            Image& img = trainImages[i / NUM_AUG];
#ifdef DUMP_TRAINING_SHAPES
            shp_i.column(0) = Regression_Shapes.column(i);
#endif
            int leaf_id_offset = 0;
            for (int l = 0; l < NUM_LANDMARKS; ++l)
            {
                float2 landmark(regShapes[2 * l][i], regShapes[2 * l + 1][i]);
                int* _leaves = nonzero_leaves + l*NUM_TREES;
                float2 rf_regressed_offset = rforests[l].run(img, landmark, _leaves); // the reached leaves, #=NUM_TREES
                                                                                      // add id offset
                for (int leaf_i = 0; leaf_i < NUM_TREES; ++leaf_i)
                    _leaves[leaf_i] += leaf_id_offset;
                leaf_id_offset += rforests[l].nLeaves;

#ifdef DUMP_TRAINING_SHAPES
                shp_i[0][l * 2] += rf_regressed_offset.x;
                shp_i[0][l * 2 + 1] += rf_regressed_offset.y;
#endif
            }
#ifdef DUMP_TRAINING_SHAPES
            {
                sprintf(tshpname, "train_smpl_rfreg_%04d_%02d_t%d.png", i / NUM_AUG, i%NUM_AUG, t);
                img.plotShape(shp_i[0]);
                img.saveAs(TRAIN_DATA_PATH + tshpname);
            }
#endif
            memcpy(Phi[i], nonzero_leaves, sizeof(int)*NUM_TREES*NUM_LANDMARKS);
        }
        printf("\n");
        toc("Run rf get local bin features");

        tic();
#pragma omp parallel for
        for (int coord_i = 0; coord_i < 2 * NUM_LANDMARKS; ++coord_i)
        {
            vector<float> reg_deltaS(Nsamples);
            float* gt_deltaS = gtOffsets[coord_i];
            if (omp_get_thread_num() == 0)
                printf("\rRegressing W %d/%d   ", coord_i + 1, 2 * NUM_LANDMARKS);

            char sbuf[256];
            sprintf(sbuf, "t%dl%d.w", t, coord_i);
            string fpath = TRAIN_DATA_PATH + sbuf;
            ridgeRegression(fpath, Nsamples, tot_leaves, gt_deltaS, Phi, LAMBDA, &reg_deltaS[0]);
            // update shape
            float *preg = regShapes[coord_i];
            for (int i = 0; i < Nsamples; ++i)
                preg[i] += reg_deltaS[i];
        }
        printf("\n");
        toc("ridge regression");

#ifdef DUMP_TRAINING_SHAPES
        printf("dumping training intermediate states: stage %d\n", t);
        for (int i = 0; i < Nsamples; ++i)
        {
            sprintf(tshpname, "train_smpl%04d_%02d_t%d.png", i / NUM_AUG, i%NUM_AUG, t);
            shp_i.column(0) = Regression_Shapes.column(i);
            trainImages[i / NUM_AUG].plotShape(shp_i[0]);
            trainImages[i / NUM_AUG].saveAs(TRAIN_DATA_PATH + tshpname);
        }
#endif
    }
}

void LBF::loadRegressor(Matrix2D<LbfRandomForest>& rforests, Matrix2D<model*>& models)
{
    printf("loading trained regressor...\n");
    string mean_shape_f = TRAIN_DATA_PATH + "meanshape";
    FILE* fp = fopen(mean_shape_f.c_str(), "rb");
    int L;
    fread(&L, sizeof(int), 1, fp);
    fread(Mean_Shape, sizeof(float), 2 * NUM_LANDMARKS, fp);
    fclose(fp);

    char fname_buf[256];
    for (int t = 1; t <= NUM_STAGES; ++t)
    {
        for (int l = 0; l < NUM_LANDMARKS; ++l)
        {
            sprintf(fname_buf, "t%dl%d.rf", t, l);
            string fpath = TRAIN_DATA_PATH + fname_buf;
            bool ret = (rforests[t - 1][l].load(fpath));
            if (!ret) { printf("error reading rf\n"); exit(-1); }

            sprintf(fname_buf, "t%dl%d.w", t, 2 * l);
            fpath = TRAIN_DATA_PATH + fname_buf;
            ret = (models[t - 1][2 * l] = load_model(fpath.c_str()));
            if (!ret) { printf("error reading W\n"); exit(-1); }

            sprintf(fname_buf, "t%dl%d.w", t, 2 * l + 1);
            fpath = TRAIN_DATA_PATH + fname_buf;
            ret = (models[t - 1][2 * l + 1] = load_model(fpath.c_str()));
            if (!ret) { printf("error reading W\n"); exit(-1); }

            printf("\r%d/%d   %d     ", t, NUM_STAGES, l);
        }
    }
    printf("\n");
}

void LBF::test()
{
    char cmd_buf[256];
    sprintf(cmd_buf, "del \"%stest_smpl*.png\"", TEST_DATA_PATH.c_str());
    printf("exec cmd...:\n\t%s\n", cmd_buf);
    system(cmd_buf);
    sprintf(cmd_buf, "del \"%stest_smpl*.shp\"", TEST_DATA_PATH.c_str());
    printf("exec cmd...:\n\t%s\n", cmd_buf);
    system(cmd_buf);

    Matrix2D<LbfRandomForest> rforests(NUM_STAGES, NUM_LANDMARKS);
    Matrix2D<model*> models(NUM_STAGES, 2 * NUM_LANDMARKS);

    loadRegressor(rforests, models);

    int nonzero_leaves[NUM_TREES*NUM_LANDMARKS];

    Image img;
    float shp[2 * NUM_LANDMARKS];
    for (int i = 0; i < TRAIN_DATA_SIZE; ++i)
    {
        string img_path = getImagePath(i);
        if (!img.load(img_path))
        {
            printf("error loading test image\n");
            exit(-1);
        }
        // reset initial shape
        memcpy(shp, Mean_Shape, sizeof(float) * 2 * NUM_LANDMARKS);
        char tshpname[256];
        sprintf(tshpname, "test_smpl%04d_t%02d.png", i, 0);
        img.plotShape(shp);
        img.saveAs(TEST_DATA_PATH + tshpname);
        sprintf(tshpname, "test_smpl%04d_t%02d.shp", i, 0);
        dumpShape(TEST_DATA_PATH + tshpname, shp);

        tic();
        for (int t = 1; t <= NUM_STAGES; ++t)
        {
            // regress local binary features
            int leaf_id_offset = 0;
            for (int l = 0; l < NUM_LANDMARKS; ++l)
            {
                int* _leaves = nonzero_leaves + l*NUM_TREES;
                rforests[t - 1][l].run(img, float2(shp[l + l], shp[2 * l + 1]), _leaves);

                // id across all forests
                for (int leaf_i = 0; leaf_i < NUM_TREES; ++leaf_i)
                    _leaves[leaf_i] += leaf_id_offset;
                leaf_id_offset += rforests[t - 1][l].nLeaves;
            }
            feature_node* feas = globalFeatures(nonzero_leaves, NUM_TREES*NUM_LANDMARKS);
            for (int coord_i = 0; coord_i < 2 * NUM_LANDMARKS; ++coord_i)
            {
                float delta = (float)predict(models[t - 1][coord_i], feas);
                shp[coord_i] += delta;
            }

            delete[] feas;

            sprintf(tshpname, "test_smpl%04d_t%02d.png", i, t);
            img.plotShape(shp);
            img.saveAs(TEST_DATA_PATH + tshpname);
            sprintf(tshpname, "test_smpl%04d_t%02d.shp", i, t);
            dumpShape(TEST_DATA_PATH + tshpname, shp);
        }
        toc();
    }
}

void LBF::loadShape(const string & fpath, int width, int height, float * shp)
{
    int dot = fpath.find_last_of('.');
    if (dot < 0) { printf("error loading shape\n"); exit(-1); }

    string ext = fpath.substr(dot + 1);
    if (ext == "land")
    {
        FILE* fp = fopen(fpath.c_str(), "rt");
        int nL;
        fscanf(fp, "%d", &nL);
        for (int i = 0; i < 2 * NUM_LANDMARKS; ++i)
        {
            float f;
            fscanf(fp, "%f", &f);
            if ((i & 1) == 0) shp[i] = f * width;
            else shp[i] = (1.f - f) * height;
        }
        fclose(fp);
    }
    else if (ext == "pts")
    {
        FILE* fp = fopen(fpath.c_str(), "rt");
        char buf[256];
        fgets(buf, 256, fp); fgets(buf, 256, fp); fgets(buf, 256, fp);
        for (int i = 0; i < 2 * NUM_LANDMARKS; ++i)
            fscanf(fp, "%f", shp + i);
        fclose(fp);
    }
}

void LBF::dumpShape(string & fpath, float * shp)
{
    FILE* fp = fopen(fpath.c_str(), "wb");
    int nLandmark = NUM_LANDMARKS;
    fwrite(&nLandmark, sizeof(int), 1, fp);
    fwrite(shp, sizeof(float), 2 * NUM_LANDMARKS, fp);
    fclose(fp);
}
