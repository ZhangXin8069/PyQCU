/* LX2 tuned STREAM-style fixed-window variant; original benchmark harness.
 * FP64 Copy z=x, Scale z=3*x, Add z=x+y, Triad z=x+3*y.
 * FMA in Triad; STREAM logical bytes 16/16/24/24 per element. */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <linux/mempolicy.h>
#include <linux/perf_event.h>
#include <omp.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/auxv.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <asm/hwcap.h>

#define MAXT 608
#define CHUNK (1024UL*1024)
#define REGS "z0","z1","z2","z3","z4","z5","z6","z7","p0"
static size_t vl,bytes;

static double seconds;
static uint64_t now_ns(clockid_t clock){struct timespec t; if(clock_gettime(clock,&t)) abort(); return (uint64_t)t.tv_sec*1000000000ULL+t.tv_nsec;}
static uint64_t wall(void){return now_ns(CLOCK_MONOTONIC_RAW);}
static void fatal(const char*s){perror(s);exit(2);}
static void pin(int cpu){cpu_set_t set;CPU_ZERO(&set);CPU_SET(cpu,&set);if(sched_setaffinity(0,sizeof(set),&set)||sched_getcpu()!=cpu)fatal("affinity");}
static long freq_khz(int cpu){char p[256];snprintf(p,sizeof(p),"/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq",cpu);FILE*f=fopen(p,"r");long n=-1;if(f){if(fscanf(f,"%ld",&n)!=1)n=-1;fclose(f);}return n;}

/* copy: one output; ordinary SVE temporal stores. */
static __attribute__((noinline)) void copy_batch(const double*a,const double*b,double*out,size_t n,long repeat){
 long loops=n/(8*vl);
 asm volatile(
 "ptrue p0.d\n"
 "2:\n"
 "mov x9,%[a]\n"
 "mov x10,%[b]\n"
 "mov x11,%[out]\n"
 "mov x12,%[loops]\n"
 "1:\n"
 "ld1d z0.d,p0/z,[x9]\n"
 "ld1d z1.d,p0/z,[x9,#1,mul vl]\n"
 "ld1d z2.d,p0/z,[x9,#2,mul vl]\n"
 "ld1d z3.d,p0/z,[x9,#3,mul vl]\n"
 "ld1d z4.d,p0/z,[x9,#4,mul vl]\n"
 "ld1d z5.d,p0/z,[x9,#5,mul vl]\n"
 "ld1d z6.d,p0/z,[x9,#6,mul vl]\n"
 "ld1d z7.d,p0/z,[x9,#7,mul vl]\n"
 "st1d z0.d,p0,[x11]\n"
 "st1d z1.d,p0,[x11,#1,mul vl]\n"
 "st1d z2.d,p0,[x11,#2,mul vl]\n"
 "st1d z3.d,p0,[x11,#3,mul vl]\n"
 "st1d z4.d,p0,[x11,#4,mul vl]\n"
 "st1d z5.d,p0,[x11,#5,mul vl]\n"
 "st1d z6.d,p0,[x11,#6,mul vl]\n"
 "st1d z7.d,p0,[x11,#7,mul vl]\n"
 "addvl x9,x9,#8\n"
 "addvl x10,x10,#8\n"
 "addvl x11,x11,#8\n"
 "subs x12,x12,#1\n"
 "b.ne 1b\n"
 "subs %[repeat],%[repeat],#1\n"
 "b.ne 2b\n"
 :[repeat]"+&r"(repeat):[a]"r"(a),[b]"r"(b),[out]"r"(out),[loops]"r"(loops):"x9","x10","x11","x12","p0","z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14","z15","z16","cc","memory");
}
/* scale: one output; ordinary SVE temporal stores. */
static __attribute__((noinline)) void scale_batch(const double*a,const double*b,double*out,size_t n,long repeat){
 long loops=n/(8*vl);
 asm volatile(
 "ptrue p0.d\n"
 "fdup z16.d,#3.0\n"
 "2:\n"
 "mov x9,%[a]\n"
 "mov x10,%[b]\n"
 "mov x11,%[out]\n"
 "mov x12,%[loops]\n"
 "1:\n"
 "ld1d z0.d,p0/z,[x9]\n"
 "ld1d z1.d,p0/z,[x9,#1,mul vl]\n"
 "ld1d z2.d,p0/z,[x9,#2,mul vl]\n"
 "ld1d z3.d,p0/z,[x9,#3,mul vl]\n"
 "ld1d z4.d,p0/z,[x9,#4,mul vl]\n"
 "ld1d z5.d,p0/z,[x9,#5,mul vl]\n"
 "ld1d z6.d,p0/z,[x9,#6,mul vl]\n"
 "ld1d z7.d,p0/z,[x9,#7,mul vl]\n"
 "fmul z0.d,p0/m,z0.d,z16.d\n"
 "fmul z1.d,p0/m,z1.d,z16.d\n"
 "fmul z2.d,p0/m,z2.d,z16.d\n"
 "fmul z3.d,p0/m,z3.d,z16.d\n"
 "fmul z4.d,p0/m,z4.d,z16.d\n"
 "fmul z5.d,p0/m,z5.d,z16.d\n"
 "fmul z6.d,p0/m,z6.d,z16.d\n"
 "fmul z7.d,p0/m,z7.d,z16.d\n"
 "st1d z0.d,p0,[x11]\n"
 "st1d z1.d,p0,[x11,#1,mul vl]\n"
 "st1d z2.d,p0,[x11,#2,mul vl]\n"
 "st1d z3.d,p0,[x11,#3,mul vl]\n"
 "st1d z4.d,p0,[x11,#4,mul vl]\n"
 "st1d z5.d,p0,[x11,#5,mul vl]\n"
 "st1d z6.d,p0,[x11,#6,mul vl]\n"
 "st1d z7.d,p0,[x11,#7,mul vl]\n"
 "addvl x9,x9,#8\n"
 "addvl x10,x10,#8\n"
 "addvl x11,x11,#8\n"
 "subs x12,x12,#1\n"
 "b.ne 1b\n"
 "subs %[repeat],%[repeat],#1\n"
 "b.ne 2b\n"
 :[repeat]"+&r"(repeat):[a]"r"(a),[b]"r"(b),[out]"r"(out),[loops]"r"(loops):"x9","x10","x11","x12","p0","z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14","z15","z16","cc","memory");
}
/* add: one output; ordinary SVE temporal stores. */
static __attribute__((noinline)) void add_batch(const double*a,const double*b,double*out,size_t n,long repeat){
 long loops=n/(8*vl);
 asm volatile(
 "ptrue p0.d\n"
 "2:\n"
 "mov x9,%[a]\n"
 "mov x10,%[b]\n"
 "mov x11,%[out]\n"
 "mov x12,%[loops]\n"
 "1:\n"
 "ld1d z0.d,p0/z,[x9]\n"
 "ld1d z1.d,p0/z,[x9,#1,mul vl]\n"
 "ld1d z2.d,p0/z,[x9,#2,mul vl]\n"
 "ld1d z3.d,p0/z,[x9,#3,mul vl]\n"
 "ld1d z4.d,p0/z,[x9,#4,mul vl]\n"
 "ld1d z5.d,p0/z,[x9,#5,mul vl]\n"
 "ld1d z6.d,p0/z,[x9,#6,mul vl]\n"
 "ld1d z7.d,p0/z,[x9,#7,mul vl]\n"
 "ld1d z8.d,p0/z,[x10]\n"
 "ld1d z9.d,p0/z,[x10,#1,mul vl]\n"
 "ld1d z10.d,p0/z,[x10,#2,mul vl]\n"
 "ld1d z11.d,p0/z,[x10,#3,mul vl]\n"
 "ld1d z12.d,p0/z,[x10,#4,mul vl]\n"
 "ld1d z13.d,p0/z,[x10,#5,mul vl]\n"
 "ld1d z14.d,p0/z,[x10,#6,mul vl]\n"
 "ld1d z15.d,p0/z,[x10,#7,mul vl]\n"
 "fadd z0.d,p0/m,z0.d,z8.d\n"
 "fadd z1.d,p0/m,z1.d,z9.d\n"
 "fadd z2.d,p0/m,z2.d,z10.d\n"
 "fadd z3.d,p0/m,z3.d,z11.d\n"
 "fadd z4.d,p0/m,z4.d,z12.d\n"
 "fadd z5.d,p0/m,z5.d,z13.d\n"
 "fadd z6.d,p0/m,z6.d,z14.d\n"
 "fadd z7.d,p0/m,z7.d,z15.d\n"
 "st1d z0.d,p0,[x11]\n"
 "st1d z1.d,p0,[x11,#1,mul vl]\n"
 "st1d z2.d,p0,[x11,#2,mul vl]\n"
 "st1d z3.d,p0,[x11,#3,mul vl]\n"
 "st1d z4.d,p0,[x11,#4,mul vl]\n"
 "st1d z5.d,p0,[x11,#5,mul vl]\n"
 "st1d z6.d,p0,[x11,#6,mul vl]\n"
 "st1d z7.d,p0,[x11,#7,mul vl]\n"
 "addvl x9,x9,#8\n"
 "addvl x10,x10,#8\n"
 "addvl x11,x11,#8\n"
 "subs x12,x12,#1\n"
 "b.ne 1b\n"
 "subs %[repeat],%[repeat],#1\n"
 "b.ne 2b\n"
 :[repeat]"+&r"(repeat):[a]"r"(a),[b]"r"(b),[out]"r"(out),[loops]"r"(loops):"x9","x10","x11","x12","p0","z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14","z15","z16","cc","memory");
}
/* triad: one output; ordinary SVE temporal stores. */
static __attribute__((noinline)) void triad_batch(const double*a,const double*b,double*out,size_t n,long repeat){
 long loops=n/(8*vl);
 asm volatile(
 "ptrue p0.d\n"
 "fdup z16.d,#3.0\n"
 "2:\n"
 "mov x9,%[a]\n"
 "mov x10,%[b]\n"
 "mov x11,%[out]\n"
 "mov x12,%[loops]\n"
 "1:\n"
 "ld1d z0.d,p0/z,[x9]\n"
 "ld1d z1.d,p0/z,[x9,#1,mul vl]\n"
 "ld1d z2.d,p0/z,[x9,#2,mul vl]\n"
 "ld1d z3.d,p0/z,[x9,#3,mul vl]\n"
 "ld1d z4.d,p0/z,[x9,#4,mul vl]\n"
 "ld1d z5.d,p0/z,[x9,#5,mul vl]\n"
 "ld1d z6.d,p0/z,[x9,#6,mul vl]\n"
 "ld1d z7.d,p0/z,[x9,#7,mul vl]\n"
 "ld1d z8.d,p0/z,[x10]\n"
 "ld1d z9.d,p0/z,[x10,#1,mul vl]\n"
 "ld1d z10.d,p0/z,[x10,#2,mul vl]\n"
 "ld1d z11.d,p0/z,[x10,#3,mul vl]\n"
 "ld1d z12.d,p0/z,[x10,#4,mul vl]\n"
 "ld1d z13.d,p0/z,[x10,#5,mul vl]\n"
 "ld1d z14.d,p0/z,[x10,#6,mul vl]\n"
 "ld1d z15.d,p0/z,[x10,#7,mul vl]\n"
 "fmla z0.d,p0/m,z8.d,z16.d\n"
 "fmla z1.d,p0/m,z9.d,z16.d\n"
 "fmla z2.d,p0/m,z10.d,z16.d\n"
 "fmla z3.d,p0/m,z11.d,z16.d\n"
 "fmla z4.d,p0/m,z12.d,z16.d\n"
 "fmla z5.d,p0/m,z13.d,z16.d\n"
 "fmla z6.d,p0/m,z14.d,z16.d\n"
 "fmla z7.d,p0/m,z15.d,z16.d\n"
 "st1d z0.d,p0,[x11]\n"
 "st1d z1.d,p0,[x11,#1,mul vl]\n"
 "st1d z2.d,p0,[x11,#2,mul vl]\n"
 "st1d z3.d,p0,[x11,#3,mul vl]\n"
 "st1d z4.d,p0,[x11,#4,mul vl]\n"
 "st1d z5.d,p0,[x11,#5,mul vl]\n"
 "st1d z6.d,p0,[x11,#6,mul vl]\n"
 "st1d z7.d,p0,[x11,#7,mul vl]\n"
 "addvl x9,x9,#8\n"
 "addvl x10,x10,#8\n"
 "addvl x11,x11,#8\n"
 "subs x12,x12,#1\n"
 "b.ne 1b\n"
 "subs %[repeat],%[repeat],#1\n"
 "b.ne 2b\n"
 :[repeat]"+&r"(repeat):[a]"r"(a),[b]"r"(b),[out]"r"(out),[loops]"r"(loops):"x9","x10","x11","x12","p0","z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14","z15","z16","cc","memory");
}
static double*alloc_on(int node){
 size_t align=2UL*1024*1024,total=bytes+align;
 void*raw=mmap(NULL,total,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);if(raw==MAP_FAILED)fatal("mmap");
 uintptr_t p=((uintptr_t)raw+align-1)&~(align-1);
 size_t before=p-(uintptr_t)raw,after=total-before-bytes;
 if(before)munmap(raw,before);if(after)munmap((void*)(p+bytes),after);
 unsigned long mask=1UL<<node;
 if(syscall(SYS_mbind,(void*)p,bytes,MPOL_BIND,&mask,64,0))fatal("mbind");
 if(madvise((void*)p,bytes,MADV_HUGEPAGE))fatal("madvise");
 return (double*)p;
}
/* Query every base-page address. Never infer placement from mbind success alone. */
static long verify_pages(double*p,int node){
 size_t page=sysconf(_SC_PAGESIZE),count=bytes/page;long checked=0;
 for(size_t off=0;off<count;off+=256){
  unsigned n=count-off<256?count-off:256;void*addr[256];int status[256];
  for(unsigned j=0;j<n;j++)addr[j]=(char*)p+(off+j)*page;
  long rc=syscall(SYS_move_pages,0,n,addr,NULL,status,0);
  if(rc<0)fatal("move_pages(query)");
  for(unsigned j=0;j<n;j++){if(status[j]!=node){fprintf(stderr,"PLACEMENT_FAIL want=%d got=%d\n",node,status[j]);exit(3);}checked++;}
 }
 return checked;
}
static int perf_open(void){struct perf_event_attr pe;memset(&pe,0,sizeof(pe));pe.type=PERF_TYPE_HARDWARE;pe.size=sizeof(pe);pe.config=PERF_COUNT_HW_CPU_CYCLES;pe.disabled=1;pe.exclude_kernel=1;pe.exclude_hv=1;pe.read_format=PERF_FORMAT_TOTAL_TIME_ENABLED|PERF_FORMAT_TOTAL_TIME_RUNNING;return syscall(SYS_perf_event_open,&pe,0,-1,-1,0);}

typedef struct{int cpu,na,nb,nout,group;double*a,*b,*out;uint64_t batches,cycles,cpu_ns,enabled,running,lag,overrun;int pmu_errno,correct;long f0,f1;long invol,vol;} Worker;
static Worker w[MAXT];
static uint64_t begin,end;
static const char*opnames[]={"copy","scale","add","triad"};
static void(*kernels[])(const double*,const double*,double*,size_t,long)={copy_batch,scale_batch,add_batch,triad_batch};
static double avalue(int t,size_t i){return 1.0+(t%13)+(i%251)*0.25;}
static double bvalue(int t,size_t i){return 2.0+(t%7)+(i%127)*0.5;}
static double expected(int op,int t,size_t i){double a=avalue(t,i),b=bvalue(t,i);return op==0?a:(op==1?3*a:(op==2?a+b:a+3*b));}
static void execute(const char*name,int count,int rounds){
 long pages=0;size_t chunk=bytes<CHUNK?bytes:CHUNK;long repeat=CHUNK/chunk;
 #pragma omp parallel num_threads(count) reduction(+:pages)
 {
  int t=omp_get_thread_num();Worker*p=&w[t];pin(p->cpu);if(omp_get_num_threads()!=count)abort();
  size_t v;asm volatile("cntb %0":"=r"(v));if(v!=vl)abort();
  p->a=alloc_on(p->na);p->b=alloc_on(p->nb);p->out=alloc_on(p->nout);
  for(size_t i=0;i<bytes/8;i++){p->a[i]=avalue(t,i);p->b[i]=bvalue(t,i);p->out[i]=-1;}
  pages+=verify_pages(p->a,p->na)+verify_pages(p->b,p->nb)+verify_pages(p->out,p->nout);
 }
 printf("{\"type\":\"placement\",\"case\":\"%s\",\"phase\":\"before\",\"correct\":true,\"verified_base_pages\":%ld}\n",name,pages);fflush(stdout);
 for(int round=0;round<rounds;round++)for(int order=0;order<4;order++){
  int op=round%2?3-order:order;
  #pragma omp parallel num_threads(count) shared(begin,end)
  {
   int t=omp_get_thread_num();Worker*p=&w[t];pin(p->cpu);p->batches=p->cycles=p->cpu_ns=p->enabled=p->running=p->lag=p->overrun=0;p->correct=1;
   for(size_t off=0;off<bytes;off+=chunk)kernels[op]((double*)((char*)p->a+off),(double*)((char*)p->b+off),(double*)((char*)p->out+off),chunk,1);
   int fd=perf_open();p->pmu_errno=fd<0?errno:0;p->f0=freq_khz(p->cpu);
   #pragma omp barrier
   #pragma omp master
   {begin=wall()+(count>72?100000000ULL:10000000ULL);end=begin+(uint64_t)(seconds*1e9);}
   #pragma omp barrier
   while(wall()<begin)asm volatile("yield");
   uint64_t t0=wall(),cpu0=now_ns(CLOCK_THREAD_CPUTIME_ID);p->lag=t0-begin;struct rusage u0,u1;getrusage(RUSAGE_THREAD,&u0);
   if(fd>=0&&(ioctl(fd,PERF_EVENT_IOC_RESET,0)||ioctl(fd,PERF_EVENT_IOC_ENABLE,0))){p->pmu_errno=errno;close(fd);fd=-1;}
   size_t offset=0;
   while(wall()<end){
    kernels[op]((double*)((char*)p->a+offset),(double*)((char*)p->b+offset),(double*)((char*)p->out+offset),chunk,repeat);
    if(wall()<=end)p->batches++;
    offset=(offset+chunk)%bytes;
   }
   uint64_t finish=wall();p->overrun=finish>end?finish-end:0;
   if(fd>=0){uint64_t v[3]={0};if(ioctl(fd,PERF_EVENT_IOC_DISABLE,0)||read(fd,v,sizeof(v))!=sizeof(v))p->pmu_errno=errno?errno:EIO;else{p->cycles=v[0];p->enabled=v[1];p->running=v[2];}close(fd);}
   p->cpu_ns=now_ns(CLOCK_THREAD_CPUTIME_ID)-cpu0;getrusage(RUSAGE_THREAD,&u1);p->invol=u1.ru_nivcsw-u0.ru_nivcsw;p->vol=u1.ru_nvcsw-u0.ru_nvcsw;p->f1=freq_khz(p->cpu);
   p->correct=(sched_getcpu()==p->cpu);
   for(size_t i=0;i<bytes/8;i++)if(p->out[i]!=expected(op,t,i)){p->correct=0;break;}
  }
  uint64_t payload=0;int good=1;for(int t=0;t<count;t++){payload+=w[t].batches*CHUNK;good &= w[t].correct;}
  printf("{\"type\":\"sample\",\"case\":\"%s\",\"kernel\":\"%s\",\"op\":%d,\"round\":%d,\"array_bytes_per_worker\":%zu,\"window_start_ns\":%"PRIu64",\"window_end_ns\":%"PRIu64",\"logical_bytes\":%"PRIu64",\"flops\":%"PRIu64",\"correct\":%s,\"workers\":[",name,opnames[op],op,round,bytes,begin,end,payload*(op<2?2:3),payload/8*(op==0?0:(op==3?2:1)),good?"true":"false");
  for(int t=0;t<count;t++){Worker*p=&w[t];uint64_t q=p->batches*CHUNK;printf("%s{\"cpu\":%d,\"a_node\":%d,\"b_node\":%d,\"out_node\":%d,\"group\":%d,\"a_read_bytes\":%"PRIu64",\"b_read_bytes\":%"PRIu64",\"write_bytes\":%"PRIu64",\"cycles_user\":%"PRIu64",\"cpu_time_ns\":%"PRIu64",\"time_enabled_ns\":%"PRIu64",\"time_running_ns\":%"PRIu64",\"pmu_errno\":%d,\"start_lag_ns\":%"PRIu64",\"overrun_ns\":%"PRIu64",\"sysfs_before_khz\":%ld,\"sysfs_after_khz\":%ld,\"involuntary_switches\":%ld,\"voluntary_switches\":%ld}",t?",":"",p->cpu,p->na,p->nb,p->nout,p->group,q,op>=2?q:0,q,p->cycles,p->cpu_ns,p->enabled,p->running,p->pmu_errno,p->lag,p->overrun,p->f0,p->f1,p->invol,p->vol);}
  puts("]}");fflush(stdout);if(!good)exit(4);
 }
 pages=0;
 #pragma omp parallel num_threads(count) reduction(+:pages)
 {Worker*p=&w[omp_get_thread_num()];pin(p->cpu);pages+=verify_pages(p->a,p->na)+verify_pages(p->b,p->nb)+verify_pages(p->out,p->nout);munmap(p->a,bytes);munmap(p->b,bytes);munmap(p->out,bytes);}
 printf("{\"type\":\"placement\",\"case\":\"%s\",\"phase\":\"after\",\"correct\":true,\"verified_base_pages\":%ld}\n",name,pages);fflush(stdout);
}
int main(int argc,char**argv){
 if(argc!=4){fprintf(stderr,"usage: stream_probe cases.txt seconds rounds\n");return 2;}seconds=atof(argv[2]);int rounds=atoi(argv[3]);if(seconds<.1||seconds>2||rounds<1||rounds>20)return 2;
 asm volatile("cntb %0":"=r"(vl));if(vl!=64)return 2;omp_set_dynamic(0);cpu_set_t allowed;sched_getaffinity(0,sizeof(allowed),&allowed);char host[256];gethostname(host,sizeof(host));
 printf("{\"type\":\"metadata\",\"version\":\"lx2-stream-v2\",\"host\":\"%s\",\"label\":\"tuned STREAM-style fixed-window variant\",\"dtype\":\"FP64\",\"sve_vl_bytes\":%zu,\"page_bytes\":%ld,\"batch_payload_bytes\":%lu,\"rounds\":%d,\"seconds\":%.9f,\"triad_fma\":true,\"prefetch_state\":\"unknown_unmodified\",\"store_policy\":\"SVE temporal\",\"hugepage_advice\":\"MADV_HUGEPAGE\"}\n",host,vl,sysconf(_SC_PAGESIZE),CHUNK,rounds,seconds);fflush(stdout);
 FILE*f=fopen(argv[1],"r");if(!f)fatal("cases");char name[128];int count,ncase=0;
 while(fscanf(f,"%127s %d %zu",name,&count,&bytes)==3){
  if(count<1||count>576||bytes<4096||bytes%4096||bytes%(8*vl)||(bytes<CHUNK&&CHUNK%bytes)||(bytes>=CHUNK&&bytes%CHUNK))return 2;
  for(int t=0;t<count;t++){Worker*p=&w[t];memset(p,0,sizeof(*p));if(fscanf(f,"%d %d %d %d %d",&p->cpu,&p->na,&p->nb,&p->nout,&p->group)!=5)return 2;
   if(p->cpu<0||p->cpu>=608||!CPU_ISSET(p->cpu,&allowed)||p->na<0||p->na>31||p->nb<0||p->nb>31||p->nout<0||p->nout>31||p->group<0||p->group>15)return 2;
   for(int j=0;j<t;j++)if(w[j].cpu==p->cpu)return 2;}
  execute(name,count,rounds);ncase++;
 }
 if(!feof(f))return 2;fclose(f);printf("{\"type\":\"done\",\"configurations\":%d,\"correct\":true}\n",ncase);return 0;
}
