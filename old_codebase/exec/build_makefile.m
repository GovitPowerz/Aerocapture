function build_makefile(machine,option)

exec = '../exec/';
dossier = '../fortran/';
results = '../objets/';

if (strcmp(machine,'sun'))
    if (strcmp(option,'optim'))
        options = '-fast -dalign -Bstatic -u';
    else
        options = '-g';
    end
    eval(['cd ' dossier]);
    !dlbk
    ddd=dir;
    eval(['cd ' exec]);

    fid = fopen('Makefile','w');
    fprintf(fid,'%s\n',['REPERF = ' dossier]);
    fprintf(fid,'%s\n','REPER = ./');
    fprintf(fid,'%s\n',['OBDIR = ' results]);
    fprintf(fid,'%s\n',['EXECU = ' exec]);
    fprintf(fid,'%s\n',' ');
    fprintf(fid,'%s\n',['FFLAGS = ' options]);
    fprintf(fid,'%s\n',' ');
    fprintf(fid,'%s','OBJETS = ');
    bla = 1:size(ddd,1);
    for kfile=bla
        if ddd(kfile).isdir==0
            file_in = ddd(kfile).name;
            iii = findstr(file_in,'.f');
            file_out = file_in;
            file_out(iii:iii+1) = '.o';
            fprintf(fid,'%s\n',['    $(OBDIR)' file_out '    \']);
        end
    end
    fprintf(fid,'%s\n',' ');
    fprintf(fid,'%s\n','PROGS = $(EXECU)Aerocap');
    fprintf(fid,'%s\n',' ');
    fprintf(fid,'%s\n','all : $(PROGS)');
    fprintf(fid,'%s\n',' ');
    fprintf(fid,'%s\n','FORCE : clean all');
    fprintf(fid,'%s\n',' ');
    fprintf(fid,'%s\n','clean :');
    fprintf(fid,'\t%s\n','rm -f $(OBJETS) $(PROGS)');
    fprintf(fid,'%s\n',' ');
    fprintf(fid,'%s\n','$(EXECU)Aerocap : $(OBJETS) $(REPER)Makefile');
    fprintf(fid,'\t%s\n','f90 $(FFLAGS) $(OBJETS) $(IMSL) -o $@');
    fprintf(fid,'%s\n',' ');
    fprintf(fid,'%s\n',['$(OBJETS) : $(REPERF)$$(@F:.o=.f) $(REPER)Makefile']);
    fprintf(fid,'\t%s\n',['f90 $(FFLAGS) -c $(REPERF)$(@F:.o=.f) -o $@']);
    fprintf(fid,'%s\n',' ');
    fclose(fid);
elseif (strcmp(machine,'linux'))
    if (strcmp(option,'optim'))
        options = '-fast -u';
    else
        options = '-g -K';
    end
    eval(['cd ' dossier]);
    !dlbk
    ddd=dir;
    eval(['cd ' exec]);

    fid = fopen('Makefile','w');
    fprintf(fid,'%s\n',['REPERF = ' dossier]);
    fprintf(fid,'%s\n',['OBDIR = ' results]);
    fprintf(fid,'%s\n',['EXECU = ' exec]);
    fprintf(fid,'%s\n',' ');
    fprintf(fid,'%s\n',['FFLAGS = ' options]);
    fprintf(fid,'%s\n',' ');
    fprintf(fid,'%s','OBJETS = ');
    bla = 1:size(ddd,1);
    for kfile=bla
        if ddd(kfile).isdir==0
            file_in = ddd(kfile).name;
            iii = findstr(file_in,'.f');
            file_out = file_in;
            file_out(iii:iii+1) = '.o';
            fprintf(fid,'\t%s\n',['$(OBDIR)' file_out '  \']);
        end
    end
    fprintf(fid,'%s\n',' ');
    fprintf(fid,'%s\n','PROGS = $(EXECU)Aerocap');
    fprintf(fid,'%s\n',' ');
    fprintf(fid,'%s\n','all : $(PROGS)');
    fprintf(fid,'%s\n',' ');
    fprintf(fid,'%s\n','$(EXECU)Aerocap : $(OBJETS)');
    fprintf(fid,'\t%s\n','f90 $(FFLAGS) $(OBJETS) $(LIB_IMSL64) -o $@');
    fprintf(fid,'%s\n',' ');
    fprintf(fid,'%s\n',['$(OBJETS) :']);
    fprintf(fid,'\t%s\n',['f90 $(FFLAGS) -c $(REPERF)$(@F:.o=.f) -o $@']);
    fprintf(fid,'%s\n',' ');
    fprintf(fid,'%s\n','clean :');
    fprintf(fid,'\t%s\n','rm -f $(OBJETS) $(PROGS)');
    fprintf(fid,'%s\n',' ');
    fprintf(fid,'%s\n','force : clean all');
    fprintf(fid,'%s\n',' ');
    fclose(fid);
end
