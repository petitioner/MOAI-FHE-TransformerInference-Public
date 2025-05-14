using namespace std;
using namespace seal;


vector<Ciphertext> ct_ct_matrix_mul_colpacking(const vector<Ciphertext> & enc_X, 
  const vector<Ciphertext> & enc_W,  const GaloisKeys & RotK, const RelinKeys &relin_keys,
  const SEALContext& seal_context, int col_X, int row_X, int col_W, int row_W, int num_batch){

  vector<Ciphertext> output(row_X);
  double scale = enc_X[0].scale();

  if(col_X != col_W || row_X != row_W){
    cout <<"ERROR: bad dimensions of X or W. "<<endl;
    return output;
  }

  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);

  

  #pragma omp parallel for 

  for (int i = 0 ; i < row_X ; ++i){
    vector<Ciphertext> copy_w(col_X);
    for (int j = 0; j < col_X; ++j){
      copy_w[j] = enc_W[j];
      if(i > 0){
        evaluator.rotate_vector_inplace(copy_w[j], i*num_batch, RotK);
      }
    }
    evaluator.multiply(enc_X[0],copy_w[0],output[i]);
    //evaluator.relinearize_inplace(output[i],relin_keys);
    //evaluator.rescale_to_next_inplace(output[i]);

    for(int j = 1 ; j < col_X ; ++j){
      Ciphertext temp;
      evaluator.multiply(enc_X[j], copy_w[j], temp);
      //evaluator.relinearize_inplace(temp,relin_keys);
      //evaluator.rescale_to_next_inplace(temp);
      evaluator.add_inplace(output[i],temp);
    }

    // put the relinearization and rescale to the end of sum. 
    evaluator.relinearize_inplace(output[i],relin_keys);
    evaluator.rescale_to_next_inplace(output[i]);
    output[i].scale()=scale;

    

  }

  return output;

}

vector<Ciphertext> ct_ct_matrix_mul_diagpacking(const vector<Ciphertext> & enc_X, 
  const vector<Ciphertext> & enc_W,  const GaloisKeys & RotK, const RelinKeys &relin_keys,
  const SEALContext& seal_context, int col_X, int row_X, int col_W, int row_W, int num_batch){

  //X: diag encoding
  //W: column encoding

  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  double scale = enc_X[0].scale();

  vector<Ciphertext> output(col_W);

  int g = sqrt((double)col_X);
  if(g * g < col_X){
    g ++;
  }

  int b = col_X/g;
  if(b * g < col_X){
    b++;
  }

  //rotate X
  
  vector<Ciphertext> rot_enc_X(row_X);

  #pragma omp parallel for 
  
  for (int i = 0; i < b; ++i){
    for (int j = 0 ; j < g ; ++j){
      int index = i*g+j;
      if(index >= row_X){
        break;
      }
      else{
        int rot_ind = (col_X-i*g)*num_batch;
        if(rot_ind != col_X*num_batch){
          evaluator.rotate_vector(enc_X[index],rot_ind,RotK,rot_enc_X[index]);
        }
        else{
          rot_enc_X[index] = enc_X[index];
        }
      }
    }
  }

  //baby step + gaint step (col_w times)
  #pragma omp parallel for 

  for (int i = 0 ; i < col_W ; ++i){
    //baby step
    vector<Ciphertext> c_g(g,enc_W[i]);

    for (int j = 1 ; j < g ; ++j){
      evaluator.rotate_vector_inplace(c_g[j], j*num_batch, RotK);
    }
    //cout <<"baby step. "<<endl;

    //gaint step
    vector<Ciphertext> out(b);
    for(int j = 0 ; j < b ; ++j){
     // cout <<"j = "<<j<<endl;
      for(int k = 0 ; k < g ; ++k){
        int index = j*g+k;
        if(index >= col_X){
          break;
        }
        if(k == 0){
          evaluator.multiply(c_g[k], rot_enc_X[index], out[j]);
          //evaluator.relinearize_inplace(out[j],relin_keys);
          //evaluator.rescale_to_next_inplace(out[j]);
        }
        else{
          Ciphertext temp;
          evaluator.multiply(c_g[k], rot_enc_X[index], temp);
          //evaluator.relinearize_inplace(temp,relin_keys);
          //evaluator.rescale_to_next_inplace(temp);
          evaluator.add_inplace(out[j],temp);
        }
      }
      evaluator.relinearize_inplace(out[j],relin_keys);
      evaluator.rescale_to_next_inplace(out[j]);
      out[j].scale()=scale;
    }
    for(int j = 0 ; j < b ; ++j){
      if (j == 0){
        output[i] = out[j];
      }
      else{
        evaluator.rotate_vector_inplace(out[j], j*g*num_batch, RotK);
        evaluator.add_inplace(output[i],out[j]);
      }
    }
  }

  return output;


}