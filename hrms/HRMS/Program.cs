using System.Text;
using Hangfire;
using Hangfire.Annotations;
using Hangfire.Dashboard;
using Hangfire.MemoryStorage;
using HRMS.Filters;
using HRMS.Helper;
using HRMS.Models.Settings;
using HRMS.Repositories;
using HRMS.Repositories.Interfaces;
using HRMS.Scheduler;
using HRMS.Services;
using HRMS.Services.Interfaces;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.OpenApi.Any;
using Microsoft.OpenApi.Models;
using NLog.Web;

internal class Program
{
    private static void Main(string[] args)
    {
        var builder = WebApplication.CreateBuilder(args);

        builder.Services.AddControllers(options =>
        {
            options.Filters.Add(typeof(LogFilter));
            options.Filters.Add(typeof(ExceptionFilter));
        }).AddNewtonsoftJson();

        builder.Services.TryAddSingleton<ILoggerService, LoggerService>();
        builder.Services.TryAddSingleton<IMemberService, MemberService>();
        builder.Services.TryAddSingleton<IBackOfficeService, BackOfficeService>();
        builder.Services.TryAddSingleton<IJwtService, JwtService>();
        builder.Services.TryAddSingleton<ISha512Service, Sha512Service>();
        builder.Services.TryAddSingleton<IHttpContextAccessor, HttpContextAccessor>();
        builder.Services.TryAddSingleton<IWorkingDateService, WorkingDateService>();
        builder.Services.TryAddSingleton<IMemberDataService, MemberDataService>();
        builder.Services.TryAddSingleton<IAuthService, AuthService>();
        builder.Services.TryAddSingleton<INotificationService, NotificationService>();
        builder.Services.TryAddSingleton<ISlackService, SlackService>();
        builder.Services.TryAddSingleton<IMemberRepository, MemberRepository>();
        builder.Services.TryAddSingleton<IBackOfficeRepository, BackOfficeRepository>();
        builder.Services.TryAddSingleton<IImageService,ImageService>();

        builder.Services.TryAddSingleton<ExceptionFilter>();
        builder.Services.TryAddSingleton<AuthenticateJwt>();
        builder.Services.TryAddSingleton<LogFilter>();
        builder.Services.TryAddSingleton<AdminPermissionFilter>();
        builder.Services.TryAddSingleton<HRPermissionFilter>();

        builder.Services.Configure<AppSettings>(builder.Configuration.GetSection("AppSettings"));
        builder.Services.TryAddSingleton(builder.Configuration.GetSection("CloudinarySettings").Get<CloudinarySettings>());
        builder.Services.Configure<BackOfficeStoreProcedureSettings>(builder.Configuration.GetSection("BackOfficeStoreProcedureSettings"));
        builder.Services.Configure<MemberStoreProcedureSettings>(builder.Configuration.GetSection("MemberStoreProcedureSettings"));

        builder.Services.AddEndpointsApiExplorer();
        var timeZoneOffset = TimeZoneInfo.Local.GetUtcOffset(DateTime.Now);
        builder.Services.AddSwaggerGen(c =>
        {
            c.SwaggerDoc("v1", new OpenApiInfo { Title = "HRMS API", Version = "v1" });
            c.MapType<TimeSpan>(() => new OpenApiSchema
            {
                Type = "string",
                Example = new OpenApiString((timeZoneOffset < TimeSpan.Zero ? "-" : "") + timeZoneOffset.ToString(@"hh\:mm"))
            });
            c.MapType<DateTimeOffset>(() => new OpenApiSchema
            {
                Type = "string",
                Example = new OpenApiString(DateTimeOffset.Now.ToString("yyyy-MM-ddTHH:mm:sszzz"))
            });
            c.AddSecurityDefinition("Bearer", new OpenApiSecurityScheme
            {
                In = ParameterLocation.Header,
                Description = "Please enter a valid token",
                Name = "Authorization",
                Type = SecuritySchemeType.Http,
                BearerFormat = "JWT",
                Scheme = "Bearer"
            });
             
            c.AddSecurityRequirement(new OpenApiSecurityRequirement
            {
                {
                    new OpenApiSecurityScheme
                    {
                        Reference = new OpenApiReference
                        {
                            Type=ReferenceType.SecurityScheme,
                            Id="Bearer"
                        }
                    },
                    new string[]{}
                }
            });
            
        });

        builder.Services.AddSwaggerGenNewtonsoftSupport();
        builder.Services.AddHttpClient<IHttpCallingHelper, HttpCallingHelper>(p => p.Timeout = TimeSpan.FromSeconds(30)).SetHandlerLifetime(TimeSpan.FromSeconds(10));
        builder.Host.UseNLog();
        builder.Services.AddHangfire(config =>
        {
            config.UseMemoryStorage();
        });
        builder.Services.AddHangfireServer();

        var app = builder.Build();

        app.UseHangfireDashboard(
            pathMatch: "/hangfire"
        );

        RecurringJob.AddOrUpdate<SendMessageScheduler>(msg => msg.RunJob(), Cron.Minutely);
        RecurringJob.AddOrUpdate<LeaveAmountScheduler>(msg => msg.RunJob(), Cron.Minutely);
        RecurringJob.AddOrUpdate<SendAbsenceScheduler>(msg => msg.RunJob(), "0 9 * * 1-5", TimeZoneInfo.FindSystemTimeZoneById("SE Asia Standard Time"));

        app.UseSwagger();
        app.UseSwaggerUI(c =>
        {
            c.SwaggerEndpoint("/swagger/v1/swagger.json", "HRMS API V1");
            
        });

        app.UseCors(x => x
                        .AllowAnyMethod()
                        .AllowAnyHeader()
                        .SetIsOriginAllowed(origin => true)
                        .AllowCredentials());

        app.UseHttpsRedirection();
        app.UseAuthorization();
        app.MapControllers();

        app.Run();
    }
}