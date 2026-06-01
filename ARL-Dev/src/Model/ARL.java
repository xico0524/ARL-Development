package Model;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.StringTokenizer;

import Jama.Matrix;

public class ARL {
	
	public String inputTrainFile = null; 
	public String inputValidFile = null;
	public String inputTestFile = null;
	public String separator = null;
	
	public ArrayList<Quadrup> trainData = new ArrayList<Quadrup>();
	public ArrayList<Quadrup> validData = new ArrayList<Quadrup>();
	public ArrayList<Quadrup> testData = new ArrayList<Quadrup>();
	
	public int uNum = 0;
	public int sNum = 0;
	public int tNum = 0;
	
	public double max_Num = 0; 
	public double min_Num = 0; 
	public double nor_pNum = 0; 
	public double max_nor_Num = 0; 
	public double min_nor_Num = 0;
	
	public int trainDataNum = 0;
	public int validDataNum = 0;
	public int testDataNum = 0;
	
	public double[][] U;
	public double[][] S;
	public double[][] T;
	
	public double[][] U_dev;
	public double[][] S_dev;
	public double[][] T_dev;
	
	public double[][] U_adv;
	public double[][] S_adv;
	public double[][] T_adv;
	
	public double[] everyRoundRMSE;
	public double[] everyRoundMAE;
	public double[] everyRoundR2;
	
	public int trainingRound = 1000;
	public int convergenceRound = 1000;
	
	public boolean flagRMSE = false;
	public boolean flagMAE = false; 
	public boolean flagR2 = false;
	
	public double minRMSE = 100;
	public double minMAE = 100; 
	public double minR2 = -100;
	
	public int minRMSERound = 0;
	public int minMAERound = 0;
	public int minR2Round = 0;
	
	public int delayCount = 10;

	public ARL(String inputTrainFile, String inputValidFile, String inputTestFile, String separator) 
	{
		this.inputTrainFile = inputTrainFile;
		this.inputValidFile = inputValidFile;
		this.inputTestFile = inputTestFile;
		this.separator = separator;
	}
	
	
	public void initData(String inputFile,ArrayList<Quadrup> data, int T)throws IOException
	{

		
		File input = new File(inputFile);
		BufferedReader in = new BufferedReader(new FileReader(input));
		String inTemp;

		while((inTemp = in.readLine()) != null ) {
			StringTokenizer st = new StringTokenizer(inTemp,separator);

			String iTemp = null;
			if(st.hasMoreTokens())
				iTemp = st.nextToken();
			
			String jTemp = null;
			if(st.hasMoreTokens())
				jTemp = st.nextToken();
			
			String kTemp = null;
			if(st.hasMoreTokens())
				kTemp = st.nextToken();
			
			String tValueTemp = null;
			if(st.hasMoreTokens())
				tValueTemp = st.nextToken();
			
			int uID = Integer.valueOf(iTemp);
			
			double s = Double.valueOf(jTemp);
			int sID = (int) s;

			double t = Double.valueOf(kTemp); 
			int tID = (int) t;
				
			double Value = Double.valueOf(tValueTemp);

			this.uNum = (this.uNum > uID) ? this.uNum : uID;
			this.sNum = (this.sNum > sID) ? this.sNum : sID;
			this.tNum = (this.tNum > tID) ? this.tNum : tID;

			this.max_Num = (this.max_Num > Value) ? this.max_Num : Value;
			this.min_Num = (this.min_Num < Value) ? this.min_Num : Value;
			
			if(T==0) {
				this.trainDataNum++;
			}
			else {
				if(T==1) {
					this.validDataNum++;
				}
				else {
					this.testDataNum++;
				}
			}

			this.nor_pNum = Math.log(Value+1); 
			
			this.max_nor_Num = (this.max_nor_Num > this.nor_pNum) ? this.max_nor_Num : this.nor_pNum;
			this.min_nor_Num = (this.min_nor_Num < this.nor_pNum) ? this.min_nor_Num : this.nor_pNum;
			
			Quadrup qtemp = new Quadrup();
			qtemp.uID = uID;
			qtemp.sID = sID;
			qtemp.tID = tID;

			qtemp.value = this.nor_pNum;
			
			data.add(qtemp); 
		}
		in.close();
	}

	
	public double randMax = 0.5;
	public double randMin = 4.9E-324;
	public double init_lf = 1.0E-3;
	
	public void initUST(int rank)
	{
		U = new double[this.uNum][rank];
		S = new double[this.sNum][rank];
		T = new double[this.tNum][rank];
		
		Random randomu = new Random();
		for(int i=0; i<uNum; i++)
		{
			for(int r=0; r<rank; r++)
			{
				U[i][r] = randMin + randomu.nextDouble() * (randMax - randMin);
			}
		}
		
		Random randoms = new Random();
		for(int j=0; j<sNum; j++)
		{
			for(int r=0; r<rank; r++)
			{
				S[j][r] = randMin + randoms.nextDouble() * (randMax - randMin);
			}
		}
		
		Random randomt = new Random();
		for(int k=0; k<tNum; k++)
		{
			for(int r=0; r<rank; r++)
			{
				T[k][r] = randMin + randomt.nextDouble() * (randMax - randMin);
			}
		}
	}


	public double[][] UnitArray;
	
	public void UnitArray(int rank) 
	{
		UnitArray = new double[rank][rank];
		
		for(int i=0; i<rank; i++) 
		{
			UnitArray[i][i] = 1;			
		}
		
	}
	

	public Map<Integer, ArrayList<RTuple>> USlice = null;
	public Map<Integer, ArrayList<RTuple>> SSlice = null;
	public Map<Integer, ArrayList<RTuple>> TSlice = null;
		
	public void partSlice() 
	{

		USlice = new HashMap<Integer,ArrayList<RTuple>>();
		SSlice = new HashMap<Integer,ArrayList<RTuple>>();
		TSlice = new HashMap<Integer,ArrayList<RTuple>>();
		
		for (Quadrup slice1: trainData)
		{

			if(USlice.containsKey(Integer.valueOf(slice1.uID)))
			{
				RTuple rtemp = new RTuple();
				rtemp.rowID = slice1.sID;
				rtemp.colID = slice1.tID;
				rtemp.mvalue = slice1.value;

				USlice.get(Integer.valueOf(slice1.uID)).add(rtemp);	
			}else {
				ArrayList<RTuple> uSlice = new ArrayList<RTuple>();
				RTuple rtemp = new RTuple();
				rtemp.rowID = slice1.sID;
				rtemp.colID = slice1.tID;
				rtemp.mvalue = slice1.value;
				uSlice.add(rtemp);
				USlice.put(Integer.valueOf(slice1.uID),uSlice);	
			}
			
			if(SSlice.containsKey(Integer.valueOf(slice1.sID)))
			{
				RTuple rtemp = new RTuple();
				rtemp.rowID = slice1.uID;
				rtemp.colID = slice1.tID;
				rtemp.mvalue = slice1.value;
				SSlice.get(Integer.valueOf(slice1.sID)).add(rtemp);
				
			}else {
				ArrayList<RTuple> sSlice = new ArrayList<RTuple>();
				RTuple rtemp = new RTuple();
				rtemp.rowID = slice1.uID;
				rtemp.colID = slice1.tID;
				rtemp.mvalue = slice1.value;
				sSlice.add(rtemp);
				SSlice.put(Integer.valueOf(slice1.sID),sSlice);	
			}
			
			if(TSlice.containsKey(Integer.valueOf(slice1.tID)))
			{
				RTuple rtemp = new RTuple();
				rtemp.rowID = slice1.uID;
				rtemp.colID = slice1.sID;
				rtemp.mvalue = slice1.value;
				TSlice.get(Integer.valueOf(slice1.tID)).add(rtemp);
			}else {
				ArrayList<RTuple> tSlice = new ArrayList<RTuple>();
				RTuple rtemp = new RTuple();
				rtemp.rowID = slice1.uID;
				rtemp.colID = slice1.sID;
				rtemp.mvalue = slice1.value;
				tSlice.add(rtemp);
				TSlice.put(Integer.valueOf(slice1.tID),tSlice);	
			}
		}
	}

	
	public double global_miu_valid = 0;
	public double global_miu_test = 0;
	
	public void compute_ave() 
	{
		double miu_valid = 0;
		
		for(Quadrup valid : validData) 
		{
			miu_valid += valid.value;  
		}
		this.global_miu_valid = miu_valid / validDataNum;
		
		double miu_test = 0;
		for(Quadrup test : testData) 
		{
			miu_test += test.value; 
		}
		this.global_miu_test = miu_test / testDataNum;
	}
	

	public double[] para_up;
	public double[] para_down;
	
	public double[] X_best;
	public double Fitness_best = 100;
	
	public double[][] X_temp;
	public double[] Fitness_temp;
	
	public double[] Theta;
	public double z = 0.01;
	
	public double[] D;
	
	public double p = 0.5;
	
	public double eta = 0.95;
	
	public double[] C;
	public double[] C_up;
	
	public double q = 4.9E-324;
	
	public double rou = 0.5;
	
	public void init_BAS()
	{
		Fitness_temp = new double[2];
		for (int i=0; i<2; i++) 
		{
			Fitness_temp[i] = 100;
		}

		para_up = new double[3];
		para_down = new double[3];
		X_best = new double[3];
		Theta = new double[3];
		D = new double[3];
		C_up = new double[3];
		C = new double[3];
		X_temp = new double[3][2];

		para_up[0] = 1.0E-2; para_down[0] = 1.0E-10;
		para_up[1] = 5.0E-3; para_down[1] = 5.0E-4;
		para_up[2] = 5.0E-3; para_down[2] = 4.8E-3;
		
		Random random = new Random();
		
		for(int i=0; i<3; i++) 
		{
			X_best[i] = para_down[i] + random.nextDouble() * (para_up[i] - para_down[i]);
			
			Theta[i] = z * (para_up[i] - para_down[i]);
			
			D[i] = Theta[i] / p;
		}
	}


	public double[][] U_update_RMSE;
	public double[][] S_update_RMSE;
	public double[][] T_update_RMSE;
	
	public double[][] U_update_MAE;
	public double[][] S_update_MAE;
	public double[][] T_update_MAE;
	
	public double[][] U_update_R2;
	public double[][] S_update_R2;
	public double[][] T_update_R2;
	
	public double[][][] U_temp;
	public double[][][] S_temp; 
	public double[][][] T_temp; 
	
	public void train(int rank) 
	{
		U_update_RMSE = new double[this.uNum][rank];
		S_update_RMSE = new double[this.sNum][rank];
		T_update_RMSE = new double[this.tNum][rank];
		
		U_update_MAE = new double[this.uNum][rank];
		S_update_MAE = new double[this.sNum][rank];
		T_update_MAE = new double[this.tNum][rank];
		
		U_update_R2 = new double[this.uNum][rank];
		S_update_R2 = new double[this.sNum][rank];
		T_update_R2 = new double[this.tNum][rank];
		
		U_dev = new double[this.uNum][rank];
		S_dev = new double[this.sNum][rank];
		T_dev = new double[this.tNum][rank];
		
		U_adv = new double[this.uNum][rank];
		S_adv = new double[this.sNum][rank];
		T_adv = new double[this.tNum][rank];
		
		U_temp = new double[2][this.uNum][rank];
		S_temp = new double[2][this.sNum][rank];
		T_temp = new double[2][this.tNum][rank];
		
		everyRoundRMSE = new double[trainingRound+1];
		
		everyRoundMAE = new double[trainingRound+1];
		
		everyRoundR2 = new double[trainingRound+1];
		
		minRMSE = 100;
		minMAE = 100;  
		minR2 = -100;
		
		minRMSERound = 0;
		minMAERound = 0;
		minR2Round = 0;

		Matrix M_UnitArray = new Matrix(UnitArray);

		double starttime = System.currentTimeMillis();

		for(int tr=1; tr<=trainingRound; tr++) {

			double starttime1 = System.currentTimeMillis();

			Random random = new Random();

			double c_down = 0;
			for (int i=0; i<3; i++)
			{
				C_up[i] =  random.nextDouble();
				c_down += Math.pow(C_up[i], 2);
			}
			C[0] = C_up[0] / Math.sqrt(c_down);
			C[1] = C_up[1] / Math.sqrt(c_down);
			C[2] = C_up[2] / Math.sqrt(c_down);

			X_temp[0][0] = X_best[0] + (D[0] * C[0])/2;
			X_temp[1][0] = X_best[1] + (D[1] * C[1])/2;
			X_temp[2][0] = X_best[2] + (D[2] * C[2])/2;
			
			X_temp[0][1] = X_best[0] - (D[0] * C[0])/2;
			X_temp[1][1] = X_best[1] - (D[1] * C[1])/2;
			X_temp[2][1] = X_best[2] - (D[2] * C[2])/2;

			for(int NP=0; NP<2; NP++)
			{
				for(Quadrup train : trainData) 
				{
					double ytemp = 0;
					double err = 0;
					
					double adv_down_u = 0;
					double adv_down_s = 0;
					double adv_down_t = 0;

					for(int r=0; r<rank; r++)
					{
						ytemp += U[train.uID-1][r] * S[train.sID-1][r] * T[train.tID-1][r];		
					}
					err = train.value - ytemp;

					for(int r=0; r<rank; r++)
					{
						U_dev[train.uID-1][r] = -err *  S[train.sID-1][r] * T[train.tID-1][r];		
						S_dev[train.sID-1][r] = -err *  U[train.uID-1][r] * T[train.tID-1][r];
						T_dev[train.tID-1][r] = -err *  U[train.uID-1][r] * S[train.sID-1][r];						
					}

					for(int r=0; r<rank; r++)
					{
						adv_down_u += Math.pow(U_dev[train.uID-1][r], 2);
						adv_down_s += Math.pow(S_dev[train.sID-1][r], 2);
						adv_down_t += Math.pow(T_dev[train.tID-1][r], 2);
					}
					adv_down_u = Math.sqrt(adv_down_u);
					adv_down_s = Math.sqrt(adv_down_s);
					adv_down_t = Math.sqrt(adv_down_t);
					
					adv_down_u += 4.9E-324;
					adv_down_s += 4.9E-324;
					adv_down_t += 4.9E-324;

					for(int r=0; r<rank; r++) 
					{
						U_adv[train.uID-1][r] = (X_temp[0][NP] * U_dev[train.uID-1][r]) / adv_down_u;
						S_adv[train.sID-1][r] = (X_temp[0][NP] * S_dev[train.sID-1][r]) / adv_down_s;
						T_adv[train.tID-1][r] = (X_temp[0][NP] * T_dev[train.tID-1][r]) / adv_down_t;
					}
				}

				for(int i=0; i<USlice.size(); i++) 
				{	
					int U_key = (int) USlice.keySet().toArray()[i];
					
					int U_key_length = USlice.get(U_key).size();

					double [] temp_Y1_real_value = new double[U_key_length];

					double [][] temp_T1 = new double[U_key_length][rank];
					double [][] temp_S1 = new double[U_key_length][rank];

					double [][] temp_T1_adv = new double[U_key_length][rank];
					double [][] temp_S1_adv = new double[U_key_length][rank];

					for (int j=0; j<U_key_length;j++) 
					{
						temp_Y1_real_value[j] = USlice.get(U_key).get(j).mvalue;

						temp_T1[j] = T[(USlice.get(U_key).get(j).colID)-1];
						temp_S1[j] = S[USlice.get(U_key).get(j).rowID-1];
						
						temp_T1_adv[j] = T_adv[(USlice.get(U_key).get(j).colID)-1];
						temp_S1_adv[j] = S_adv[USlice.get(U_key).get(j).rowID-1];
						
					}

					Matrix M_temp_Y1_real_value = new Matrix(temp_Y1_real_value,1);

					Matrix M_ui_adv = new Matrix(U_adv[i],1);
					
					Matrix M_temp_T1 = new Matrix(temp_T1);
					Matrix M_temp_S1 = new Matrix(temp_S1);
					
					Matrix M_temp_T1_adv = new Matrix(temp_T1_adv);
					Matrix M_temp_S1_adv = new Matrix(temp_S1_adv);

					Matrix M_temp_T1_adv_add = M_temp_T1.plus(M_temp_T1_adv);
					Matrix M_temp_S1_adv_add = M_temp_S1.plus(M_temp_S1_adv);

					Matrix M_temp_adv_add_Had1 = M_temp_T1_adv_add.arrayTimes(M_temp_S1_adv_add);

					Matrix M_temp_adv_add_Had1_trans = M_temp_adv_add_Had1.transpose();

					Matrix M_U_left_1 = M_temp_Y1_real_value.times(M_temp_T1.arrayTimes(M_temp_S1));

					Matrix M_U_left_2 = M_temp_Y1_real_value.times(M_temp_adv_add_Had1.times(X_temp[1][NP]));

					Matrix M_U_left_3 = M_ui_adv.times(M_temp_adv_add_Had1_trans.times(M_temp_adv_add_Had1.times(-X_temp[1][NP])));

					Matrix M_U_left_all = M_U_left_1.plus(M_U_left_2.plus(M_U_left_3));

					Matrix M_U_right_1 = (M_temp_T1.arrayTimes(M_temp_S1).transpose()).times(M_temp_T1.arrayTimes(M_temp_S1));

					Matrix M_U_right_2 = M_UnitArray.times(U_key_length * X_temp[2][NP]);

					Matrix M_U_right_3 = M_temp_adv_add_Had1_trans.times(M_temp_adv_add_Had1.times(X_temp[1][NP]));

					Matrix M_U_right_all_inverse = (M_U_right_1.plus(M_U_right_2.plus(M_U_right_3))).inverse();

					Matrix M_U_all = M_U_left_all.times(M_U_right_all_inverse);

					U[U_key-1] = M_U_all.getArray()[0];
				}				

				
				for(int i=0; i<SSlice.size(); i++)
				{	
					int S_key = (int) SSlice.keySet().toArray()[i];

					int S_key_length = SSlice.get(S_key).size();

					double [] temp_Y2_real_value = new double[S_key_length];

					double [][] temp_T2 = new double[S_key_length][rank];
					double [][] temp_U2 = new double[S_key_length][rank];

					double [][] temp_T2_adv = new double[S_key_length][rank];
					double [][] temp_U2_adv = new double[S_key_length][rank];

					for (int j=0; j<S_key_length;j++) 
					{
						temp_Y2_real_value[j] = SSlice.get(S_key).get(j).mvalue;

						temp_T2[j] = T[(SSlice.get(S_key).get(j).colID)-1];
						temp_U2[j] = U[SSlice.get(S_key).get(j).rowID-1];			
						
						temp_T2_adv[j] = T_adv[(SSlice.get(S_key).get(j).colID)-1];
						temp_U2_adv[j] = U_adv[SSlice.get(S_key).get(j).rowID-1];
						
					}

					Matrix M_temp_Y2_real_value = new Matrix(temp_Y2_real_value,1);

					Matrix M_sj_adv = new Matrix(S_adv[i],1);
					
					Matrix M_temp_T2 = new Matrix(temp_T2);
					Matrix M_temp_U2 = new Matrix(temp_U2);
					
					Matrix M_temp_T2_adv = new Matrix(temp_T2_adv);
					Matrix M_temp_U2_adv = new Matrix(temp_U2_adv);

					Matrix M_temp_T2_adv_add = M_temp_T2.plus(M_temp_T2_adv);
					Matrix M_temp_U2_adv_add = M_temp_U2.plus(M_temp_U2_adv);

					Matrix M_temp_adv_add_Had2 = M_temp_T2_adv_add.arrayTimes(M_temp_U2_adv_add);

					Matrix M_temp_adv_add_Had2_trans = M_temp_adv_add_Had2.transpose();

					Matrix M_S_left_1 = M_temp_Y2_real_value.times(M_temp_T2.arrayTimes(M_temp_U2));

					Matrix M_S_left_2 = M_temp_Y2_real_value.times(M_temp_adv_add_Had2.times(X_temp[1][NP]));

					Matrix M_S_left_3 = M_sj_adv.times(M_temp_adv_add_Had2_trans.times(M_temp_adv_add_Had2.times(-X_temp[1][NP])));

					Matrix M_S_left_all = M_S_left_1.plus(M_S_left_2.plus(M_S_left_3));

					Matrix M_S_right_1 = (M_temp_T2.arrayTimes(M_temp_U2).transpose()).times(M_temp_T2.arrayTimes(M_temp_U2));

					Matrix M_S_right_2 = M_UnitArray.times(S_key_length * X_temp[2][NP]);

					Matrix M_S_right_3 = M_temp_adv_add_Had2_trans.times(M_temp_adv_add_Had2.times(X_temp[1][NP]));

					Matrix M_S_right_all_inverse = (M_S_right_1.plus(M_S_right_2.plus(M_S_right_3))).inverse();

					Matrix M_S_all = M_S_left_all.times(M_S_right_all_inverse);

					S[S_key-1] = M_S_all.getArray()[0];
				}


				for(int i=0; i<TSlice.size(); i++)
				{
					int T_key = (int) TSlice.keySet().toArray()[i];

					int T_key_length = TSlice.get(T_key).size();

					double [] temp_Y3_real_value = new double[T_key_length];

					double [][] temp_S3 = new double[T_key_length][rank];
					double [][] temp_U3 = new double[T_key_length][rank];

					double [][] temp_S3_adv = new double[T_key_length][rank];
					double [][] temp_U3_adv = new double[T_key_length][rank];

					for (int j=0; j<T_key_length;j++) 
					{
						temp_Y3_real_value[j] = TSlice.get(T_key).get(j).mvalue;

						temp_S3[j] = S[(TSlice.get(T_key).get(j).colID)-1];
						temp_U3[j] = U[TSlice.get(T_key).get(j).rowID-1];			
						
						temp_S3_adv[j] = S_adv[(TSlice.get(T_key).get(j).colID)-1];
						temp_U3_adv[j] = U_adv[TSlice.get(T_key).get(j).rowID-1];
						
					}

					Matrix M_temp_Y3_real_value = new Matrix(temp_Y3_real_value,1);	

					Matrix M_tk_adv = new Matrix(T_adv[i],1);

					Matrix M_temp_S3 = new Matrix(temp_S3);
					Matrix M_temp_U3 = new Matrix(temp_U3);
					
					Matrix M_temp_S3_adv = new Matrix(temp_S3_adv);
					Matrix M_temp_U3_adv = new Matrix(temp_U3_adv);

					Matrix M_temp_S3_adv_add = M_temp_S3.plus(M_temp_S3_adv);
					Matrix M_temp_U3_adv_add = M_temp_U3.plus(M_temp_U3_adv);

					Matrix M_temp_adv_add_Had3 = M_temp_S3_adv_add.arrayTimes(M_temp_U3_adv_add);

					Matrix M_temp_adv_add_Had3_trans = M_temp_adv_add_Had3.transpose();

					Matrix M_T_left_1 = M_temp_Y3_real_value.times(M_temp_S3.arrayTimes(M_temp_U3));

					Matrix M_T_left_2 = M_temp_Y3_real_value.times(M_temp_adv_add_Had3.times(X_temp[1][NP]));

					Matrix M_T_left_3 = M_tk_adv.times(M_temp_adv_add_Had3_trans.times(M_temp_adv_add_Had3.times(-X_temp[1][NP])));

					Matrix M_T_left_all = M_T_left_1.plus(M_T_left_2.plus(M_T_left_3));

					Matrix M_T_right_1 = (M_temp_S3.arrayTimes(M_temp_U3).transpose()).times(M_temp_S3.arrayTimes(M_temp_U3));

					Matrix M_T_right_2 = M_UnitArray.times(T_key_length * X_temp[2][NP]);

					Matrix M_T_right_3 = M_temp_adv_add_Had3_trans.times(M_temp_adv_add_Had3.times(X_temp[1][NP]));

					Matrix M_T_right_all_inverse = (M_T_right_1.plus(M_T_right_2.plus(M_T_right_3))).inverse();

					Matrix M_T_all = M_T_left_all.times(M_T_right_all_inverse);

					T[T_key-1] = M_T_all.getArray()[0];
				}
				

				for(int r=0; r<rank;r++) 
				{
					for(int i=0; i<uNum; i++) 
					{
						U_temp[NP][i][r] = U[i][r];
					}
					
					for(int j=0; j<sNum; j++) 
					{
						S_temp[NP][j][r] = S[j][r];
					}
					
					for(int k=0; k<tNum; k++) 
					{
						T_temp[NP][k][r] = T[k][r];
					}
				}
				
			}

			
			double RMSEUp_train_left = 0; 
			double MAEUp_train_left = 0;
			
			double RMSEUp_train_right = 0; 
			double MAEUp_train_right = 0;
			
			for(Quadrup train : trainData) 
			{
				double ytemp_left = 0;
				double ytemp_right = 0;
				for(int yr=0; yr<rank; yr++)
				{
					ytemp_left += U_temp[0][train.uID-1][yr] * S_temp[0][train.sID-1][yr] * T_temp[0][train.tID-1][yr];
					ytemp_right += U_temp[1][train.uID-1][yr] * S_temp[1][train.sID-1][yr] * T_temp[1][train.tID-1][yr];
				}
				
				RMSEUp_train_left += Math.pow(train.value - ytemp_left, 2);
				MAEUp_train_left += Math.abs(train.value - ytemp_left);
				
				RMSEUp_train_right += Math.pow(train.value - ytemp_right, 2);
				MAEUp_train_right += Math.abs(train.value - ytemp_right);
			}
			
			double RMSE_train_left = Math.sqrt(RMSEUp_train_left / trainDataNum);
			double MAE_train_left = MAEUp_train_left / trainDataNum;
			
			double RMSE_train_right = Math.sqrt(RMSEUp_train_right / trainDataNum);
			double MAE_train_right = MAEUp_train_right / trainDataNum;
			
			Fitness_temp[0] = (rou * RMSE_train_left) + ((1-rou) * MAE_train_left);
			Fitness_temp[1] = (rou * RMSE_train_right) + ((1-rou) * MAE_train_right);
			
			
			if( Fitness_temp[0] < Fitness_temp[1]) 
			{
				Fitness_best = Fitness_temp[0];
				
				for(int num=0; num<3; num++)
				{
					if(X_best[num] < para_down[num]) 
					{
						X_best[num] = para_down[num];
					}
					else if(X_best[num] > para_up[num]) 
					{
						X_best[num] = para_up[num];
					}
					else
					{
						X_best[num] = X_temp[num][0];
					}
					
				}
				
				for(int r=0; r<rank;r++)
				{
					for(int i=0; i<uNum; i++) 
					{
						U[i][r] = U_temp[0][i][r];
					}
					for(int j=0; j<sNum; j++)
					{
						S[j][r] = S_temp[0][j][r];
					}
					for(int k=0; k<tNum; k++) 
					{
						T[k][r] = T_temp[0][k][r];
					}
				}
				
			}
			
			else
			{
				Fitness_best = Fitness_temp[1];
				
				for(int num=0; num<3; num++)
				{
					if(X_best[num] < para_down[num]) 
					{
						X_best[num] = para_down[num];
					}
					else if(X_best[num] > para_up[num]) 
					{
						X_best[num] = para_up[num];
					}
					else
					{
						X_best[num] = X_temp[num][1];
					}
					
				}
				
				for(int r=0; r<rank;r++)
				{
					for(int i=0; i<uNum; i++) 
					{
						U[i][r] = U_temp[1][i][r];
					}
					for(int j=0; j<sNum; j++)
					{
						S[j][r] = S_temp[1][j][r];
					}
					for(int k=0; k<tNum; k++) 
					{
						T[k][r] = T_temp[1][k][r];
					}
				}
			}

			for(int a=0; a<3; a++)
			{
				Theta[a] = eta*Theta[a] + q;
				D[a] = eta*D[a] + q;
			}
			
			double RMSEUp = 0; 
			double MAEUp = 0;
			
			double R2Up = 0;
			double R2Down = 0;
			
			for(Quadrup valid : validData) 
			{
				double ytemp = 0;
				for(int yr=0; yr<rank; yr++)
				{
					ytemp += U[valid.uID-1][yr] * S[valid.sID-1][yr] * T[valid.tID-1][yr];
				}
				
				RMSEUp += Math.pow(valid.value - ytemp, 2);
				MAEUp += Math.abs(valid.value - ytemp);	
				
				R2Up += Math.pow(valid.value - ytemp, 2);
				R2Down += Math.pow(valid.value - this.global_miu_valid, 2);	
			}
			
			everyRoundRMSE[tr] = Math.sqrt(RMSEUp/validDataNum);
			everyRoundMAE[tr] = MAEUp / validDataNum;
			everyRoundR2[tr] = 1 - ( (R2Up) / (R2Down) );

			if((Math.abs(everyRoundRMSE[tr]-minRMSE) >= 0.0001) && (everyRoundRMSE[tr] < minRMSE))
			{
				minRMSE = everyRoundRMSE[tr];
				minRMSERound = tr;
				
				for (int r=0; r<rank;r++)
				{
					for(int i=0; i<uNum; i++) 
					{
						U_update_RMSE[i][r] = U[i][r];
					}
					for(int j=0; j<sNum; j++) 
					{
						S_update_RMSE[j][r] = S[j][r];
					}
					for(int k=0; k<tNum; k++) 
					{
						T_update_RMSE[k][r] = T[k][r];
					}
				}		
			}
			else
			{
				if((tr - minRMSERound) >= delayCount) 
				{
					flagRMSE = true;
					if(flagMAE && flagR2) 
					{
						convergenceRound = tr;
						break;
					}
				}
			}

			if((Math.abs(everyRoundMAE[tr]-minMAE) >= 0.0001) && (everyRoundMAE[tr] < minMAE))
			{
				minMAE = everyRoundMAE[tr];
				minMAERound = tr;
				
				for (int r=0; r<rank;r++)
				{
					for(int i=0; i<uNum; i++) 
					{
						U_update_MAE[i][r] = U[i][r];
					}
					for(int j=0; j<sNum; j++) 
					{
						S_update_MAE[j][r] = S[j][r];
					}
					for(int k=0; k<tNum; k++) 
					{
						T_update_MAE[k][r] = T[k][r];
					}
				}
			}
			else
			{
				if((tr - minMAERound) >= delayCount) 
				{
					flagMAE = true;
					if(flagRMSE && flagR2)
					{
						convergenceRound = tr;
						break;
					}
				}
			}

			if((Math.abs(minR2-everyRoundR2[tr]) >= 0.0001) && (everyRoundR2[tr] > minR2))
			{
				minR2 = everyRoundR2[tr];
				minR2Round = tr;
				
				for (int r=0; r<rank;r++)
				{
					for(int i=0; i<uNum; i++) 
					{
						U_update_R2[i][r] = U[i][r];
					}
					for(int j=0; j<sNum; j++) 
					{
						S_update_R2[j][r] = S[j][r];
					}
					for(int k=0; k<tNum; k++) 
					{
						T_update_R2[k][r] = T[k][r];
					}
				}
			}
			else
			{
				if((tr - minR2Round) >= delayCount) 
				{
					flagR2 = true;
					if(flagRMSE && flagMAE)
					{
						convergenceRound = tr;
						break;
					}
				}
			}
			
			double endtime1 = System.currentTimeMillis();
			System.out.println("Current RMSE (Validation set): "+minRMSE+", minRMSERound: "+minRMSERound);
			System.out.println("Current MAE (Validation set): "+minMAE+", minMAERound: "+minMAERound);
			System.out.println("Current R2 (Validation set): "+minR2+", minR2Round: "+minR2Round);
			System.out.println("Current round time: "+(endtime1-starttime1)/1000+" seconds \n");
		}
		
		System.out.println("\nConvergenceRound: "+convergenceRound);
		System.out.println("Minimum RMSE (Validation set): "+minRMSE+", minRMSERound: "+minRMSERound);
		System.out.println("Minimum MAE(Validation set): "+minMAE+", minMAERound: "+minMAERound);
		System.out.println("Minimum R2(Validation set): "+minR2+", minR2Round: "+minR2Round);				

		
		double RMSEUp_final = 0; 
		double MAEUp_final = 0;
		
		double R2Up_final = 0;
		double R2Down_final = 0;
		
		for (Quadrup test: testData)
		{
			double ytemp_RMSE = 0;
			double ytemp_MAE = 0;
			double ytemp_R2 = 0;
			
			for(int yr=0; yr<rank; yr++)
			{
				ytemp_RMSE += U_update_RMSE[test.uID-1][yr] * S_update_RMSE[test.sID-1][yr] * T_update_RMSE[test.tID-1][yr]; 
				ytemp_MAE += U_update_MAE[test.uID-1][yr] * S_update_MAE[test.sID-1][yr] * T_update_MAE[test.tID-1][yr];
				ytemp_R2 += U_update_R2[test.uID-1][yr] * S_update_R2[test.sID-1][yr] * T_update_R2[test.tID-1][yr];
			}
			
			RMSEUp_final += Math.pow(test.value- ytemp_RMSE, 2);
			MAEUp_final += Math.abs(test.value - ytemp_MAE);	
			
			R2Up_final += Math.pow(test.value - ytemp_R2, 2);
			R2Down_final += Math.pow(test.value - this.global_miu_test, 2);
		}
		
		double RMSE_final = Math.sqrt(RMSEUp_final / testDataNum);
		double MAE_final = MAEUp_final / testDataNum;
		double R2_final = 1 - (R2Up_final / R2Down_final);
		
		System.out.println("\nMinimum RMSE (Testing set): " + RMSE_final + ", minRMSERound: "+minRMSERound);
		System.out.println("Minimum MAE (Testing set): " + MAE_final + ", minMAERound: "+minMAERound);
		System.out.println("Minimum R2 (Testing set): " + R2_final + ", minR2Round: "+minR2Round);
		
		double endtime = System.currentTimeMillis();
		System.out.println("Training time cost: "+(endtime-starttime)/1000+" seconds\n");
	}
	

	public static void main(String[] args)throws NumberFormatException,IOException,InterruptedException {    
		
		System.out.println("Start");
		System.out.println("************************************");
		double all_start = System.currentTimeMillis();
		
		ARL cpals = new ARL(
				"./Samples/Training.txt",
				"./Samples/Validation.txt",
				"./Samples/Testing.txt",
				":");
		
		try {
			cpals.initData(cpals.inputTrainFile,cpals.trainData, 0);
			cpals.initData(cpals.inputValidFile,cpals.validData, 1);
			cpals.initData(cpals.inputTestFile,cpals.testData, 2);
			
			int r = 20;
			
			for(int i=0; i<1; i++)
			{
				cpals.initUST(r);
				cpals.UnitArray(r);
				cpals.partSlice();
				
				cpals.compute_ave();
				
				cpals.init_BAS();
				
				cpals.train(r);
			}
			
		} catch(IOException e) {
			e.printStackTrace();		
		}
		
		double all_end = System.currentTimeMillis();
		System.out.println("\nTotal time cost:"+(all_end-all_start)/1000+" seconds");
		System.out.println("************************************");
		System.out.println("End");
	}
}
