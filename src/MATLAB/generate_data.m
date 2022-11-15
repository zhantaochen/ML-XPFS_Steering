function generate_data(J1_list, DM_list, K_list, path_saving)
    num_samples = length(J1_list);
    
    J2 =  0.0;
    J3 =  0.0;
    Jc = -0.6;
    G  =  0;
    Dz = -0.1;
    
    for i_sample = 1: num_samples
    
        J1 = J1_list(i_sample);
        DM = DM_list(i_sample);
        K = K_list(i_sample);
    
        params.J1 = J1;
        params.J2 = J2;
        params.J3 = J3;
        params.DM = DM;
        params.Dz = Dz;
        params.Jc = Jc;
        params.K = K;
        params.G = G;cryst = spinw;
        cryst.genlattice( ...
            'lat_const',[6.867, 6.867, 19.807],'angled',[90, 90, 120],'spgr',148)
        cryst.addatom('r', [1/3 2/3 1/3], 'S', 3/2, 'label', 'Cr1', 'color', 'b')
        cryst.gencoupling('maxDistance', 10)
    
        if abs(J1) > 0
            cryst.addmatrix('label', 'J1', 'color', 'r', 'value', J1)
            cryst.addcoupling('mat', 'J1','bond', 1)
        end
    
        if abs(J2) > 0
            cryst.addmatrix('label', 'J2', 'color', 'g', 'value', J2)
            cryst.addcoupling('mat', 'J2','bond', 3)
        end
    
        if abs(J3) > 0
            cryst.addmatrix('label', 'J3', 'color', 'b', 'value', J3)
            % here the J3 actually corresponds to 4th nearest neighbor
            % since it is longer than inter-plane bonds
            cryst.addcoupling('mat', 'J3','bond', 6)
        end
    
        if abs(Jc) > 0
            disp('adding inter-layer interaction')
            cryst.addmatrix('label', 'Jc', 'color', 'orange', 'value', Jc)
            % adding inter-layer interaction
            cryst.addcoupling('mat', 'Jc','bond', 2)
        end
    
        % adding DMI
        if abs(DM) > 0
            disp('adding DMI')
            % value and orientation will be defined later for the DMI
            cryst.addmatrix('label', 'DM', 'color', 'magenta', 'value', [0,0,1] * DM)
            cryst.addcoupling('mat', 'DM','bond', 3)
            % value and orientation will be defined later for the DMI
            %     cryst.addmatrix('label', 'DM', 'color', 'magenta', 'value', DM)
            %     cryst.addcoupling('mat', 'DM', 'bond', 2)
            %     cryst.setmatrix('mat', 'DM', 'pref', {[0 0 DM]})
        end
    
        % % adding anisotropy
        if abs(Dz) > 0
            disp('adding ANISOTROPY')
            cryst.addmatrix('label', 'Dz', 'color', 'gray', 'value', diag([0 0 Dz]))
            cryst.addaniso('Dz')
        end
    
        if abs(K) > 0
            disp('adding KITAEV')
            gamma_z = [sqrt(2)/sqrt(3) 0 1/sqrt(3)];
            gamma_x = (get_inplane_rotmat(2/3 * pi) * gamma_z')';
            gamma_y = (get_inplane_rotmat(4/3 * pi) * gamma_z')';
            cryst.addmatrix('label', 'Jxx', 'color', 'r', 'value', K * (gamma_x' * gamma_x))
            cryst.addmatrix('label', 'Jyy', 'color', 'g', 'value', K * (gamma_y' * gamma_y))
            cryst.addmatrix('label', 'Jzz', 'color', 'b', 'value', K * (gamma_z' * gamma_z))
            cryst.addcoupling('mat','Jxx','bond',1,'subidx',[4 5 6]);
            cryst.addcoupling('mat','Jyy','bond',1,'subidx',[7 8 9]);
            cryst.addcoupling('mat','Jzz','bond',1,'subidx',[1 2 3]);
        end
    
        if abs(G) > 0
            cryst.addmatrix('label', 'Gyz', 'color', 'r', ...
                'value', [0 0 0; 0 0 G; 0 G 0])
            cryst.addmatrix('label', 'Gzx', 'color', 'r', ...
                'value', [0 0 G; 0 0 0; G 0 0])
            cryst.addmatrix('label', 'Gxy', 'color', 'r', ...
                'value', [0 G 0; G 0 0; 0 0 0])
            cryst.addcoupling('mat','Gyz','bond',1,'subidx',[1 2 3]);
            cryst.addcoupling('mat','Gzx','bond',1,'subidx',[4 5 6]);
            cryst.addcoupling('mat','Gxy','bond',1,'subidx',[7 8 9]);
        end
    
    
        cryst.genmagstr('mode', 'random');
        cryst.optmagsteep('nRun', 10000);
    
        Spec = cryst.spinwave({[1/3 1/3 0] [0 0 0] [1 0 0] 250},'hermit',false);
        %     Spec = cryst.spinwave({[1 1 0] [0 0 0] [1 0 0] 250},'hermit',false);
        Spec = sw_egrid(Spec,'component','Sxx+Syy+Szz','imagChk',false);
        Spec = sw_neutron(Spec);
    
        %     figure
        %     subplot(2,1,1)
        %     sw_plotspec(Spec,'mode',1,'axLim',[0 25],'colorbar',false',...
        %         'colormap',[0 0 0],'imag',true,'sortMode',true,'dashed',true)
        %     subplot(2,1,2)
        %     sw_plotspec(Spec,'mode',3,'dE',0.5,'axLim',[0 25],'dashed',true)
        %     swplot.subfigure(1,3,1)
    
        % saving used parameters
        struct_saving.J1 = params.J1;
        struct_saving.J2 = params.J2;
        struct_saving.J3 = params.J3;
        struct_saving.DM = params.DM;
        struct_saving.Dz = params.Dz;
        struct_saving.Jc = params.Jc;
        struct_saving.K  = params.K;
        struct_saving.G  = params.G;
        % saving spectrum results
        struct_saving.hkl    = Spec.hkl;
        struct_saving.hklA   = Spec.hklA;
        struct_saving.Evect  = Spec.Evect;
        struct_saving.swConv = Spec.swConv;
        struct_saving.omega  = Spec.omega;
    
        save([path_saving, sprintf('sample_%d.mat',i_sample)], '-struct', 'struct_saving');
    end
end