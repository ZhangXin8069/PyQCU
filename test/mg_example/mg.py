import numpy as cp
import numpy as np
import lattice
import bicgstab
# from cupyx.scipy.sparse.linalg import eigsh
# from cupyx.scipy.sparse.linalg import LinearOperator

      

class mg:

    blocksize = [4, 2, 2, 2, 2, 2] #每一层的单个方向的压缩程度
    coarse_dof = [8, 12, 12, 12, 12, 12] #新一层的内禀维度
    R_null_vec = [] #
    mg_ops = []
    coarse_map = []
    fine_sites_per_coarse_list = []

    #生成近零空间向量
    def near_null_vec(self, P_null_vec_coarse, coarse_dof, coarse_op, info = bicgstab.cg_info()):
        for i in range(0, coarse_dof):
            #施密特正交化
            for k in range(0,i):
                P_null_vec_coarse[i,:,:,:] -= cp.vdot((P_null_vec_coarse[k,:,:,:]),P_null_vec_coarse[i,:,:,:])/cp.vdot((P_null_vec_coarse[k,:,:,:]),P_null_vec_coarse[k,:,:,:])*P_null_vec_coarse[k,:,:,:]

            #Ar
            Ar = lattice.apply_mat(P_null_vec_coarse[i,:,:,:], coarse_op)
            #-Ar
            Ar = -Ar
            #x = (A^-1)*(-Ar)
            x = bicgstab.bicgstab(Ar, op=coarse_op, tol=5e-5, info=info)
            #V = x+r
            P_null_vec_coarse[i,:,:,:] += x
            # print(P_null_vec_coarse[i,:,0,0])


            #施密特正交化
            for k in range(0,i):
                P_null_vec_coarse[i,:,:,:] -= cp.vdot((P_null_vec_coarse[k,:,:,:]),P_null_vec_coarse[i,:,:,:])/cp.vdot((P_null_vec_coarse[k,:,:,:]),P_null_vec_coarse[k,:,:,:])*P_null_vec_coarse[k,:,:,:]

            # P_null_vec_coarse[i,:] = cp.zeros_like(P_null_vec_coarse[i,:])
            P_null_vec_coarse[i,:] = P_null_vec_coarse[i,:]/cp.sqrt(cp.vdot((P_null_vec_coarse[i,:]), P_null_vec_coarse[i,:]))
            # if i==0:
                # print("after",P_null_vec_coarse[i,:])
            # print(lattice.apply_mat(P_null_vec_coarse[i,:,:,:], op = coarse_op)[:,0,0])

        return P_null_vec_coarse
    
    def vol_index_dof_to_cv_index(self, i, i_dof, coarse_op):
        return coarse_op.nc*i + i_dof

    def index_to_coord(self, ptr, coarse_op):
        y_coarse = ptr % coarse_op.ny
        x_coarse = ptr // coarse_op.ny
        return x_coarse, y_coarse
    
    #网格物理地址与内存地址转化
    def coord_to_index(self, coords, coarse_op, i):
        ptr = (coarse_op.ny*coords[0] + coords[1])*coarse_op.nc + i
        return ptr

    def zeros_like_fermi(self, level):
        fermi_out = cp.random.rand(self.mg_ops[level].nx, self.mg_ops[level].ny, self.mg_ops[level].nc*2).view(cp.complex128)
        fermi_out = cp.zeros_like(fermi_out)
        return fermi_out

    '''
    建立粗网格与细网格之间的对应关系
    '''
    def build_mapping(self, map_id, fine_op, coarse_op):
        print("Buliding map...")
        for i in range(0,coarse_op.volume):
            x_coarse, y_coarse = self.index_to_coord(i, coarse_op)
            coarse_coords = [x_coarse, y_coarse]
            # print("coarse_coords = ",coarse_coords)
            coords = [0, 0]
            blocksizes = int(fine_op.nx/coarse_op.nx)
            count = [0]
            self.recursive_site_build(map_id, coarse_coords, coords, 0, count, blocksizes, fine_op, i)
        print("Buliding map finished")

    '''
    build_mapping用到的递归函数
    '''
    def recursive_site_build(self, map_id, coarse_coords, coords, step, count, blocksizes, fine_op, fine_ptr):
        if(step < 2):
            for i in range(coarse_coords[step]*blocksizes, (coarse_coords[step]+1)*blocksizes):
                coords[step] = i
                self.recursive_site_build(map_id, coarse_coords, coords, step+1, count, blocksizes, fine_op, fine_ptr)
        else:
            for i in range(0, fine_op.nc):
                self.coarse_map[map_id][fine_ptr][count[0]] = self.coord_to_index(coords, fine_op, i)
                count[0] = count[0] + 1

    def restrict_f2c(self, fine_level, fermi_in, fermi_out):
        fine_sites_per_coarse = self.fine_sites_per_coarse_list[fine_level]
        nevc = self.coarse_dof[fine_level]
        fermi_in = fermi_in.reshape(-1)
        fermi_out = fermi_out.reshape(-1)
        for i in range(0,self.mg_ops[fine_level+1].volume):
            for i_dof in range(0,nevc):
                cv_index = self.vol_index_dof_to_cv_index(i, i_dof, self.mg_ops[fine_level+1])
                for j in range(0, fine_sites_per_coarse):
                    fermi_out[cv_index] += np.conj(self.R_null_vec[fine_level][i_dof][self.coarse_map[fine_level][i][j]])*fermi_in[self.coarse_map[fine_level][i][j]]



    def prolong_c2f(self, fine_level, fermi_in, fermi_out):
        fine_sites_per_coarse = self.fine_sites_per_coarse_list[fine_level]
        nevc = self.coarse_dof[fine_level]
        fermi_in = fermi_in.reshape(-1)
        fermi_out = fermi_out.reshape(-1)
        for i in range(0,self.mg_ops[fine_level+1].volume):
            for i_dof in range(0,nevc):
                cv_index = self.vol_index_dof_to_cv_index(i, i_dof, self.mg_ops[fine_level+1])
                for j in range(0, fine_sites_per_coarse):
                    fermi_out[self.coarse_map[fine_level][i][j]] += self.R_null_vec[fine_level][i_dof][self.coarse_map[fine_level][i][j]]*fermi_in[cv_index]


    def local_orthogonalization(self, fine_level, nevc, fine_sites_per_coarse):
        for i in range(0,self.mg_ops[fine_level+1].volume):
            for i_dof in range(0,nevc):
                cv_index = self.vol_index_dof_to_cv_index(i, i_dof, self.mg_ops[fine_level+1])
                for k in range(0,i_dof):

                    k_dot = 0
                    k_i_dof_dot = 0
                    for j in range(0, fine_sites_per_coarse):
                        k_dot += np.conj(self.R_null_vec[fine_level][k][self.coarse_map[fine_level][i][j]])*self.R_null_vec[fine_level][k][self.coarse_map[fine_level][i][j]]
                        k_i_dof_dot += np.conj(self.R_null_vec[fine_level][k][self.coarse_map[fine_level][i][j]])*self.R_null_vec[fine_level][i_dof][self.coarse_map[fine_level][i][j]]
                    for j in range(0, fine_sites_per_coarse):
                        self.R_null_vec[fine_level][i_dof][self.coarse_map[fine_level][i][j]] -= self.R_null_vec[fine_level][k][self.coarse_map[fine_level][i][j]] * k_i_dof_dot / k_dot 
                    
                i_dof_dot = 0
                for j in range(0, fine_sites_per_coarse):
                    i_dof_dot += np.conj(self.R_null_vec[fine_level][i_dof][self.coarse_map[fine_level][i][j]])*self.R_null_vec[fine_level][i_dof][self.coarse_map[fine_level][i][j]]

                for j in range(0, fine_sites_per_coarse):
                    self.R_null_vec[fine_level][i_dof][self.coarse_map[fine_level][i][j]] /= cp.sqrt(i_dof_dot)
                a=0




    def fermi_f2c(self, fermi, fine_op, coarse_op):
        return 0
    

    def __init__(self, fine_op, n_refine, ifeigen=0):
        self.n_refine = n_refine
        self.fine_op = fine_op
        U = self.fine_op.U
        nx = self.fine_op.nx
        ny = self.fine_op.ny
        nc = self.fine_op.nc
        if(ifeigen == 0):
            print("搭建多重网格...")
            self.mg_ops.append(fine_op)######################################
            for i in range(0,n_refine):
                print("当前层： ", i)
                print("创造近零空间向量...")
                for s in range(10):
                    info_null_vec = bicgstab.cg_info()
                    P_null_vec_coarse = cp.random.rand(self.coarse_dof[i], nx, ny, nc*2).view(cp.complex128)
                    P_null_vec_coarse = self.near_null_vec(P_null_vec_coarse, self.coarse_dof[i], self.mg_ops[i], info=info_null_vec)
                    if (info_null_vec.if_max_iter==0):
                        break
                P_null_vec_coarse = P_null_vec_coarse.reshape(P_null_vec_coarse.shape[0],-1)
                self.R_null_vec.append(P_null_vec_coarse)######################################
                print("近零空间向量创造完毕")
                
                rand_fermi =  cp.random.rand(nx, ny, nc*2).view(cp.complex128)
                rand_fermi =  cp.zeros_like(rand_fermi)
                rand_fermi =  cp.ones_like(rand_fermi)
                # rand_fermi[0,0,0] = 1000
                # print(P_null_vec_coarse[1,:,:,:])

                nx = int(nx/self.blocksize[i])
                ny = int(ny/self.blocksize[i])
                nc_c = nc
                nc = int(self.coarse_dof[i])

                map = [[int(0)]*int(self.blocksize[i]*self.blocksize[i]*nc_c)] * int(nx*ny)
                fine_sites_per_coarse = int(self.blocksize[i]*self.blocksize[i]*nc_c)
                self.fine_sites_per_coarse_list.append(fine_sites_per_coarse)

                map = np.array(map)
                self.coarse_map.append(map)######################################         



                coarse_op = lattice.operator_para(nx, ny, nc)
                self.mg_ops.append(coarse_op)
                self.build_mapping( i, self.mg_ops[i], self.mg_ops[i+1])
                # print(self.coarse_map[i].shape)
                # print(self.coarse_map[i])
                # print(len(np.unique(self.coarse_map[i])) < len(self.coarse_map[i]))

                self.local_orthogonalization(i, self.coarse_dof[i], fine_sites_per_coarse)
                
                
                fermi_out = cp.random.rand(nx, ny, nc*2).view(cp.complex128)
                fermi_out  = self.zeros_like_fermi(i+1)
                fermi_out = cp.zeros_like(fermi_out)
                # fermi_out[0,0,0] = 1
                # fermi_out[0,1,0] = 1
                # fermi_out[1,0,0] = 1
                # fermi_out[1,1,0] = 1
                fermi_out = fermi_out
                fermi_out_pr = cp.zeros_like(fermi_out)
                fermi_out_r = cp.zeros_like(rand_fermi)
                # self.restrict_f2c( i, rand_fermi, fermi_out)
                # self.prolong_c2f( i, fermi_out, fermi_out_r)
                # self.restrict_f2c( i, fermi_out_r, fermi_out_pr)
                # print(rand_fermi)
                # print(fermi_out)
                # print(fermi_out_r)
                # print(fermi_out_pr)

                print("生成更粗一层对角元与非对角元...")
                ################################# transfer ######################################
                self.mg_ops[i+1].clover = cp.zeros_like(self.mg_ops[i+1].clover)
                # clover
                print("clover:")
                for color in range(0,self.mg_ops[i+1].nc):
                    print(color)
                    
                    fermi_tmp_coarse = cp.zeros_like(fermi_out)
                    fermi_tmp_fine = cp.zeros_like(fermi_out_r)
                    fermi_tmp_Afine = cp.zeros_like(fermi_out_r)

                    fermi_tmp_coarse[:,:,color] = 1

                    # print("fermi_tmp_coarse = ",fermi_tmp_coarse)

                    self.prolong_c2f( i, fermi_tmp_coarse, fermi_tmp_fine)
                    fermi_tmp_Afine = lattice.apply_clover(fermi_tmp_fine, self.mg_ops[i])
                    fermi_tmp_coarse = cp.zeros_like(fermi_tmp_coarse)
                    self.restrict_f2c( i, fermi_tmp_Afine, fermi_tmp_coarse)

                    self.mg_ops[i+1].clover[:,:,:,color] = fermi_tmp_coarse[:,:,:]

                    # print("fermi_tmp_coarse = ",fermi_tmp_coarse)
                    # print("clover = ", self.mg_ops[i+1].clover)

                # wilson
                # self.mg_ops[i+1].clover = cp.zeros_like(self.mg_ops[i+1].clover)
                self.mg_ops[i+1].hopping = cp.zeros_like(self.mg_ops[i+1].hopping)
                print("wilson:")
                for color in range(0,self.mg_ops[i+1].nc):
                    print(color)
                    # xp=even
                    fermi_tmp_coarse = cp.zeros_like(fermi_out)
                    fermi_tmp_fine = cp.zeros_like(fermi_out_r)
                    fermi_tmp_Afine = cp.zeros_like(fermi_out_r)

                    fermi_tmp_coarse[0:-1:2,:,color] = 1
                    self.prolong_c2f( i, fermi_tmp_coarse, fermi_tmp_fine)
                    fermi_tmp_Afine = lattice.apply_hopping_x_p(fermi_tmp_fine, self.mg_ops[i])
                    fermi_tmp_coarse = cp.zeros_like(fermi_tmp_coarse)
                    self.restrict_f2c( i, fermi_tmp_Afine, fermi_tmp_coarse)
                    self.mg_ops[i+1].hopping[0,1::2,:,:,color] = fermi_tmp_coarse[1::2,:,:]
                    self.mg_ops[i+1].clover[0:-1:2,:,:,color] += fermi_tmp_coarse[0:-1:2,:,:]
                    # print("fermi_tmp_coarse = ", fermi_tmp_coarse)

                    # xp=odd
                    fermi_tmp_coarse = cp.zeros_like(fermi_out)
                    fermi_tmp_fine = cp.zeros_like(fermi_out_r)
                    fermi_tmp_Afine = cp.zeros_like(fermi_out_r)

                    fermi_tmp_coarse[1::2,:,color] = 1
                    # print("fermi_tmp_coarse", fermi_tmp_coarse)
                    self.prolong_c2f( i, fermi_tmp_coarse, fermi_tmp_fine)
                    fermi_tmp_Afine = lattice.apply_hopping_x_p(fermi_tmp_fine, self.mg_ops[i])
                    fermi_tmp_coarse = cp.zeros_like(fermi_tmp_coarse)
                    self.restrict_f2c( i, fermi_tmp_Afine, fermi_tmp_coarse)
                    self.mg_ops[i+1].hopping[0,0:-1:2,:,:,color] = fermi_tmp_coarse[0:-1:2,:,:]
                    self.mg_ops[i+1].clover[1::2,:,:,color] += fermi_tmp_coarse[1::2,:,:]

                    # xm=even
                    fermi_tmp_coarse = cp.zeros_like(fermi_out)
                    fermi_tmp_fine = cp.zeros_like(fermi_out_r)
                    fermi_tmp_Afine = cp.zeros_like(fermi_out_r)

                    fermi_tmp_coarse[0:-1:2,:,color] = 1
                    self.prolong_c2f( i, fermi_tmp_coarse, fermi_tmp_fine)
                    fermi_tmp_Afine = lattice.apply_hopping_x_m(fermi_tmp_fine, self.mg_ops[i])
                    fermi_tmp_coarse = cp.zeros_like(fermi_tmp_coarse)
                    self.restrict_f2c( i, fermi_tmp_Afine, fermi_tmp_coarse)
                    self.mg_ops[i+1].hopping[1,1::2,:,:,color] = fermi_tmp_coarse[1::2,:,:]
                    self.mg_ops[i+1].clover[0:-1:2,:,:,color] += fermi_tmp_coarse[0:-1:2,:,:]

                    # xm=odd
                    fermi_tmp_coarse = cp.zeros_like(fermi_out)
                    fermi_tmp_fine = cp.zeros_like(fermi_out_r)
                    fermi_tmp_Afine = cp.zeros_like(fermi_out_r)

                    fermi_tmp_coarse[1::2,:,color] = 1
                    self.prolong_c2f( i, fermi_tmp_coarse, fermi_tmp_fine)
                    fermi_tmp_Afine = lattice.apply_hopping_x_m(fermi_tmp_fine, self.mg_ops[i])
                    fermi_tmp_coarse = cp.zeros_like(fermi_tmp_coarse)
                    self.restrict_f2c( i, fermi_tmp_Afine, fermi_tmp_coarse)
                    self.mg_ops[i+1].hopping[1,0:-1:2,:,:,color] = fermi_tmp_coarse[0:-1:2,:,:]
                    self.mg_ops[i+1].clover[1::2,:,:,color] += fermi_tmp_coarse[1::2,:,:]

                    # yp=even
                    fermi_tmp_coarse = cp.zeros_like(fermi_out)
                    fermi_tmp_fine = cp.zeros_like(fermi_out_r)
                    fermi_tmp_Afine = cp.zeros_like(fermi_out_r)

                    fermi_tmp_coarse[:,0:-1:2,color] = 1
                    self.prolong_c2f( i, fermi_tmp_coarse, fermi_tmp_fine)
                    fermi_tmp_Afine = lattice.apply_hopping_y_p(fermi_tmp_fine, self.mg_ops[i])
                    fermi_tmp_coarse = cp.zeros_like(fermi_tmp_coarse)
                    self.restrict_f2c( i, fermi_tmp_Afine, fermi_tmp_coarse)
                    self.mg_ops[i+1].hopping[2,:,1::2,:,color] = fermi_tmp_coarse[:,1::2,:]
                    self.mg_ops[i+1].clover[:,0:-1:2,:,color] += fermi_tmp_coarse[:,0:-1:2,:]

                    # yp=odd
                    fermi_tmp_coarse = cp.zeros_like(fermi_out)
                    fermi_tmp_fine = cp.zeros_like(fermi_out_r)
                    fermi_tmp_Afine = cp.zeros_like(fermi_out_r)

                    fermi_tmp_coarse[:,1::2,color] = 1
                    self.prolong_c2f( i, fermi_tmp_coarse, fermi_tmp_fine)
                    fermi_tmp_Afine = lattice.apply_hopping_y_p(fermi_tmp_fine, self.mg_ops[i])
                    fermi_tmp_coarse = cp.zeros_like(fermi_tmp_coarse)
                    self.restrict_f2c( i, fermi_tmp_Afine, fermi_tmp_coarse)
                    self.mg_ops[i+1].hopping[2,:,0:-1:2,:,color] = fermi_tmp_coarse[:,0:-1:2,:]
                    self.mg_ops[i+1].clover[:,1::2,:,color] += fermi_tmp_coarse[:,1::2,:]

                    # ym=even
                    fermi_tmp_coarse = cp.zeros_like(fermi_out)
                    fermi_tmp_fine = cp.zeros_like(fermi_out_r)
                    fermi_tmp_Afine = cp.zeros_like(fermi_out_r)

                    fermi_tmp_coarse[:,0:-1:2,color] = 1
                    self.prolong_c2f( i, fermi_tmp_coarse, fermi_tmp_fine)
                    fermi_tmp_Afine = lattice.apply_hopping_y_m(fermi_tmp_fine, self.mg_ops[i])
                    fermi_tmp_coarse = cp.zeros_like(fermi_tmp_coarse)
                    self.restrict_f2c( i, fermi_tmp_Afine, fermi_tmp_coarse)
                    self.mg_ops[i+1].hopping[3,:,1::2,:,color] = fermi_tmp_coarse[:,1::2,:]
                    self.mg_ops[i+1].clover[:,0:-1:2,:,color] += fermi_tmp_coarse[:,0:-1:2,:]

                    # ym=odd
                    fermi_tmp_coarse = cp.zeros_like(fermi_out)
                    fermi_tmp_fine = cp.zeros_like(fermi_out_r)
                    fermi_tmp_Afine = cp.zeros_like(fermi_out_r)

                    fermi_tmp_coarse[:,1::2,color] = 1
                    self.prolong_c2f( i, fermi_tmp_coarse, fermi_tmp_fine)
                    fermi_tmp_Afine = lattice.apply_hopping_y_m(fermi_tmp_fine, self.mg_ops[i])
                    fermi_tmp_coarse = cp.zeros_like(fermi_tmp_coarse)
                    self.restrict_f2c( i, fermi_tmp_Afine, fermi_tmp_coarse)
                    self.mg_ops[i+1].hopping[3,:,0:-1:2,:,color] = fermi_tmp_coarse[:,0:-1:2,:]
                    self.mg_ops[i+1].clover[:,1::2,:,color] += fermi_tmp_coarse[:,1::2,:]

                fermi_tmp_coarse = cp.zeros_like(fermi_out)
                fermi_tmp_fine = cp.zeros_like(fermi_out_r)
                fermi_tmp_Afine = cp.zeros_like(fermi_out_r)
                fermi_tmp_coarse_n = cp.zeros_like(fermi_out)
                fermi_tmp_coarse_t = cp.zeros_like(fermi_out)
                fermi_tmp_coarse[:,:,0] = 1

                # print("self.mg_ops[i+1].hopping", self.mg_ops[i+1].hopping[0,:,:,:,:])
                fermi_tmp_coarse_n = lattice.apply_mat(fermi_tmp_coarse,self.mg_ops[i+1])
                self.prolong_c2f( i, fermi_tmp_coarse, fermi_tmp_fine)
                fermi_tmp_Afine = lattice.apply_mat(fermi_tmp_fine, self.mg_ops[i])
                self.restrict_f2c( i, fermi_tmp_Afine, fermi_tmp_coarse_t)


                # print("clover = ", self.mg_ops[i+1].clover)
                # print("self.mg_ops[i+1].hopping", self.mg_ops[i+1].hopping[0,:,:,:,:])

                # print("fermi_tmp_coarse_n = ",fermi_tmp_coarse_n)
                # print("fermi_tmp_coarse_t = ",fermi_tmp_coarse_t)
                print("dif = ",cp.linalg.norm(fermi_tmp_coarse_n-fermi_tmp_coarse_t))

            

        else:
            a=0


    def mg_bicgstab_recursive(self, b, max_iter=300, tol=1e-10, if_info=1, info = bicgstab.cg_info(), level=0, relative_tol=0):
        x = cp.zeros_like(b)

        if level < self.n_refine+1:
            # 计算初始残差 r = b - Ax
            r = b - lattice.apply_mat(x, self.mg_ops[level])
            if relative_tol != 0:
                tol = cp.sqrt(cp.vdot(r, r))*relative_tol
            # print(r)
            r0 = r.copy()  # 保存初始残差 r0
            p = r.copy()   # 初始化搜索方向 p
            rho = 1
            rho1 = 1
            w = 1
            alpha = 1
            count = 0
            if if_info!=0:
                a = " "*(level)
                print(a, "level = ", level, cp.sqrt(cp.vdot(r, r)))
            # 主迭代循环
            for k in range(max_iter):
                count += 1
                # 计算 Ap = A * p
                Ap = lattice.apply_mat(p, self.mg_ops[level])
                
                # 计算步长 alpha
                alpha = cp.vdot((r0), r) / cp.vdot((r0), Ap)
                # print("alpha = ", alpha)
                        
                x += alpha * p

                # 更新中间残差 r_1 = r - alpha * Ap
                r_1 = r - alpha * Ap

                # 检查是否收敛
                if if_info!=0:
                    a = " "*(level)
                    print(a, "level = ", level, cp.sqrt(cp.vdot(r_1, r_1)))

                if cp.sqrt(cp.vdot(r_1, r_1)) < tol:
                    if if_info!=0:
                        a = " "*(level)
                        print(a, "level = ", level, "RelRes", cp.sqrt(cp.vdot(r_1, r_1))/cp.sqrt(cp.vdot(r0, r0)))
                        # print("count = ",count)
                    info.count = count
                    info.norm_r = cp.sqrt(cp.vdot(r_1, r_1))
                    info.r = r_1
                    return x
                
                # 计算 t = A * r
                t = lattice.apply_mat(r, self.mg_ops[level])
                
                # 计算 omega
                omega = cp.vdot((t), r) / cp.vdot((t), t)
                
                # 更新解 x
                x += omega * r_1
                
                # 更新残差 r = r_1 - omega * t
                r_1 = r_1 - omega * lattice.apply_mat(r_1, self.mg_ops[level])
                
                if level == 0:
                    a=0
                if level < self.n_refine:
                    #下潜
                    r_coarse = self.zeros_like_fermi(level=level+1)
                    # z1_prec_fine = bicgstab.bicgstab(r_1, op=self.mg_ops[level], if_info=0, relative_tol=1e-1)
                    # z1_fine = lattice.apply_mat(z1_prec_fine, self.mg_ops[level])
                    self.restrict_f2c(level, r_1, r_coarse)

                    #递归
                    info_c = bicgstab.cg_info()
                    relative_tol=0.25
                    if level+1 == self.n_refine:
                        e_coarse = self.mg_bicgstab_recursive(r_coarse, level=level+1, info=info_c, relative_tol=0.1, if_info=0, max_iter=100)
                    else:
                        e_coarse = self.mg_bicgstab_recursive(r_coarse, level=level+1, info=info_c, relative_tol=0.25, if_info=1, max_iter=100)
                    a = " "*(level+1)
                    print(a, "level", level+1, " ", "iter", info_c.count)
                    

                    #上浮
                    z2_fine = self.zeros_like_fermi(level=level)
                    e0_fine = self.zeros_like_fermi(level=level)
                    self.prolong_c2f(level, e_coarse, z2_fine)
                    # e0_fine += z1_fine
                    e0_fine += z2_fine

                    # Ap = lattice.apply_mat(e0_fine, self.mg_ops[level])
                    # alpha = cp.vdot(Ap, r_1) / cp.vdot(Ap, Ap)


                    if info_c.if_max_iter == 0:
                        x = x + e0_fine
                        r_1 = b - lattice.apply_mat(x, self.mg_ops[level])

                else:
                    a=0
                    # x_stack[-1] = x

                # 检查是否收敛
                if cp.sqrt(cp.vdot(r_1, r_1)) < tol:
                    if if_info!=0:
                        a = " "*(level)
                        print(a, "level = ", level, "RelRes", cp.sqrt(cp.vdot(r_1, r_1))/cp.sqrt(cp.vdot(r0, r0)))
                    #     print("count = ",count)
                    info.count = count
                    info.norm_r = cp.sqrt(cp.vdot(r_1, r_1))
                    info.r = r_1
                    return x
                # 计算 beta
                beta = (cp.vdot((r_1), r_1) / cp.vdot((r), r)) 
                
                # 更新搜索方向 p
                p = r_1 + alpha*beta/omega*p - alpha*beta*Ap

                r = r_1

            # 如果未收敛，抛出错误
            a = " "*(level)
            print(a, "level", level, "over max_iter")
            info.if_max_iter=1
            return x

        if level == 0:
            print("level = ",level,"   ")

    def mg_minres(self, b, max_iter=300, tol=1e-10, if_info=1, info = bicgstab.cg_info(), level=0, relative_tol=0):
        x = cp.zeros_like(b)

        if level < self.n_refine+1:
            # 计算初始残差 r = b - Ax
            r = b - lattice.apply_mat(x, self.mg_ops[level])
            if relative_tol != 0:
                tol = cp.sqrt(cp.vdot(r, r))*relative_tol
            # print(r)
            r0 = r.copy()  # 保存初始残差 r0
            alpha = 0
            count = 0
            if if_info!=0:
                a = " "*(level)
                print(a, "level = ", level, cp.sqrt(cp.vdot(r, r)))


            # 主迭代循环
            for k in range(max_iter):

                p = lattice.apply_mat(r, self.mg_ops[level])

                count += 1

                
                # 计算步长 alpha
                alpha = cp.vdot(p, r) / cp.vdot(p, p)
                # print("alpha = ", alpha)
                        
                x += alpha * r

                # 更新中间残差 r_1 = r - alpha * Ap
                r = r - alpha * p

                # 检查是否收敛
                if if_info!=0:
                    a = " "*(level)
                    print(a, "level = ", level, cp.sqrt(cp.vdot(r, r)))

                if cp.sqrt(cp.vdot(r, r)) < tol:
                    if if_info!=0:
                        a = " "*(level)
                        print(a, "level = ", level, "RelRes", cp.sqrt(cp.vdot(r, r))/cp.sqrt(cp.vdot(r0, r0)))
                        # print("count = ",count)
                    info.count = count
                    info.norm_r = cp.sqrt(cp.vdot(r, r))
                    info.r = r.copy()
                    return x
            info.if_max_iter=1
            return x



    def mg_gcr_recursive(self, b, max_iter=300, tol=1e-10, if_info=1, info = bicgstab.cg_info(), level=0, relative_tol=0):
        x = cp.zeros_like(b)

        if level < self.n_refine+1:
            # 计算初始残差 r = b - Ax
            r = b - lattice.apply_mat(x, self.mg_ops[level])
            if relative_tol != 0:
                tol = cp.sqrt(cp.vdot(r, r))*relative_tol
            # print(r)
            r0 = r.copy()  # 保存初始残差 r0
            alpha = 1
            count = 0
            if if_info!=0:
                a = " "*(level)
                print(a, "level = ", level, cp.sqrt(cp.vdot(r, r)))
            
            p_store = []
            Ap_store = []

            z1 = self.mg_minres(r, max_iter=2, tol=0, if_info=0,  level=level)
            r1 = r-lattice.apply_mat(z1, self.mg_ops[level])
            #下潜
            r_coarse = self.zeros_like_fermi(level=level+1)
            self.restrict_f2c(level, r1, r_coarse)

            #递归 z = M^(-1) r
            info_c = bicgstab.cg_info()
            relative_tol=0.25
            if level+1 == self.n_refine:
                e_coarse =  bicgstab.bicgstab(r_coarse, op=self.mg_ops[level+1], if_info=0, relative_tol=0.01, info=info_c)
            else:
                e_coarse = self.mg_gcr_recursive(r_coarse, level=level+1, info=info_c, relative_tol=0.25, if_info=1, max_iter=10)
            # e_coarse =  bicgstab.bicgstab(r_coarse, op=self.mg_ops[level+1], if_info=0, relative_tol=0.001, info=info_c)
            a = " "*(level+1)
            print(a, "level", level+1, " ", "iter", info_c.count)
            

            #上浮
            z2 = self.zeros_like_fermi(level=level)
            z_fine = self.zeros_like_fermi(level=level)
            self.prolong_c2f(level, e_coarse, z2)
            z_fine += z2 + z1
            p = z_fine.copy()
            Ap = lattice.apply_mat(p, self.mg_ops[level])

                
            
            

            # 主迭代循环
            for k in range(max_iter):

                p_store.append(p)
                Ap_store.append(Ap)

                count += 1
                # 计算 Ap = A * p
                Ap = lattice.apply_mat(p, self.mg_ops[level])
                
                # 计算步长 alpha
                alpha = cp.vdot(Ap, r) / cp.vdot(Ap, Ap)
                # print("alpha = ", alpha)
                        
                x += alpha * p

                # 更新中间残差 r_1 = r - alpha * Ap
                r = r - alpha * Ap

                # 检查是否收敛
                if if_info!=0:
                    a = " "*(level)
                    print(a, "level = ", level, cp.sqrt(cp.vdot(r, r)))

                if cp.sqrt(cp.vdot(r, r)) < tol:
                    
                    if if_info!=0:
                        a = " "*(level)
                        print(a, "level = ", level, "RelRes", cp.sqrt(cp.vdot(r, r))/cp.sqrt(cp.vdot(r0, r0)))
                        # print("count = ",count)
                    info.count = count
                    info.norm_r = cp.sqrt(cp.vdot(r, r))
                    info.r = r.copy()
                    return x
                
               
                
                if level == 0:
                    a=0
                    print("tol=",tol)
                if level < self.n_refine:
                    z1 = self.mg_minres(r, max_iter=2, tol=0, if_info=0,  level=level)
                    r1 = r-lattice.apply_mat(z1, self.mg_ops[level])

                    #下潜
                    r_coarse = self.zeros_like_fermi(level=level+1)
                    self.restrict_f2c(level, r1, r_coarse)

                    #递归 z = M^(-1) r
                    info_c = bicgstab.cg_info()
                    relative_tol=0.25
                    if level+1 == self.n_refine:
                        e_coarse =  bicgstab.bicgstab(r_coarse, op=self.mg_ops[level+1], if_info=0, relative_tol=0.01, info=info_c)
                    else:
                        e_coarse = self.mg_gcr_recursive(r_coarse, level=level+1, info=info_c, relative_tol=relative_tol, if_info=1, max_iter=10)
                    a = " "*(level+1)
                    print(a, "level", level+1, " ", "iter", info_c.count)
                    

                    #上浮
                    z2 = self.zeros_like_fermi(level=level)
                    z_fine = self.zeros_like_fermi(level=level)
                    self.prolong_c2f(level, e_coarse, z2)
                    z_fine += z2 + z1
                    Az = lattice.apply_mat(z_fine, self.mg_ops[level])

                    p[:] = z_fine[:]
                    Ap[:] = Az[:]

                    # p[:] = r[:]
                    # Ar = lattice.apply_mat(r, self.mg_ops[level])
                    # Ap[:] = Ar[:]

                    for ii in range(0,k+1):
                        beta_ij = -cp.vdot(Ap_store[ii], Az)/cp.vdot(Ap_store[ii], Ap_store[ii])
                        p += beta_ij*p_store[ii]
                        Ap += beta_ij*Ap_store[ii]



                else:
                    a=0
                    # x_stack[-1] = x



            # 如果未收敛，抛出错误
            a = " "*(level)
            print(a, "level", level, "over max_iter")
            info.if_max_iter=1
            return x

        if level == 0:
            print("level = ",level,"   ")


    def mg_bicgstab(self, b,op=0,  x0=None, max_iter=3000, tol=1e-10, if_info=0, info = bicgstab.cg_info(), relative_tol=0):
        buf = 1
        if buf==0:
            X = self.mg_bicgstab_recursive( b, max_iter=max_iter, tol=tol, info = info, level=0, relative_tol=0)
            print("mg_bicgstab_recursive.count = ", info.count)
        else:
            X = self.mg_gcr_recursive( b, max_iter=max_iter, tol=tol, info = info, level=0, relative_tol=0)
            print("mg_gcr_recursive.count = ", info.count)
        return X
    
    
        


