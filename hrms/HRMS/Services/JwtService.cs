using DeviceDetectorNET;
using HRMS.Models;
using HRMS.Models.Settings;
using HRMS.Services.Interfaces;
using Microsoft.Extensions.Options;
using Microsoft.IdentityModel.Tokens;
using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using System.Text;
using DeviceDetectorNET.Parser.Client;
using XAct;

namespace HRMS.Services
{
    public class JwtService : IJwtService
    {
        private readonly AppSettings _appSettings;
        private readonly ILoggerService _loggerService;
        private readonly IHttpContextAccessor _httpContextAccessor;

        public JwtService(IOptions<AppSettings> appSettings, ILoggerService loggerService, IHttpContextAccessor httpContextAccessor)
        {
            _appSettings = appSettings.Value;
            _loggerService = loggerService;
            _httpContextAccessor = httpContextAccessor;
        }

        public string GenerateToken(JwtData data)
        {

            try
            {
                var detector = new DeviceDetector(_httpContextAccessor.HttpContext?.Request.Headers["User-Agent"].ToString());
                detector.Parse();
                var tokenHandler = new JwtSecurityTokenHandler();
                var key = Encoding.ASCII.GetBytes(_appSettings.Secret);
                var claims = new List<Claim>
                {
                    new Claim("Id", data.Id.ToString()),
                    new Claim("Username", data.Username),
                    new Claim("Permission", data.Permission.ToString()),
                };
                claims.AddRange(data.Permissions.Select(permission => new Claim(type: "Permissions", permission)));
                var tokenDescriptor = new SecurityTokenDescriptor
                {
                    Subject = new ClaimsIdentity(claims: claims),
                    SigningCredentials = new SigningCredentials(new SymmetricSecurityKey(key), SecurityAlgorithms.HmacSha256Signature),
                    // Expires = DateTime.UtcNow.AddHours(1),
                };

                 if (!detector.GetClient().ParserName.ToString().Contains("library"))
                 {
                     tokenDescriptor.Expires = DateTime.UtcNow.AddHours(1);
                 }

                _loggerService.Info($"GenerateToken UserAgent: {detector.GetClient()}");
                var token = tokenHandler.CreateToken(tokenDescriptor);
                return tokenHandler.WriteToken(token);
            }
            catch (Exception e)
            {
                _loggerService.Error($"GenerateToken Exception: {e.Message}");
                return null;
            }
        }

        public JwtData DecryptToken(string token, bool validateLifetime = false)
        {
            try
            {
                var detector = new DeviceDetector(_httpContextAccessor.HttpContext?.Request.Headers["User-Agent"].ToString());
                detector.Parse();
                var tokenHandler = new JwtSecurityTokenHandler();
                var key = Encoding.ASCII.GetBytes(_appSettings.Secret);
                tokenHandler.ValidateToken(token, new TokenValidationParameters
                {
                    ValidateIssuerSigningKey = true,
                    ValidateIssuer = false,
                    ValidateAudience = false,
                    ValidateLifetime = !detector.GetClient().ParserName.ToString().Contains("library"),
                    ClockSkew = TimeSpan.Zero,
                    IssuerSigningKey = new SymmetricSecurityKey(key),
                }, out var validatedToken);
                var jwtToken = (JwtSecurityToken)validatedToken;
                if (!jwtToken.Claims.Any()) return null;
                
                
                _loggerService.Info($"DecryptToken UserAgent: {detector.GetClient()}");

                return new JwtData
                {
                    Id = int.Parse(jwtToken.Claims.First(x => x.Type == "Id").Value),
                    Username = jwtToken.Claims.First(x => x.Type == "Username").Value,
                    Permission = int.Parse(jwtToken.Claims.First(x => x.Type == "Permission").Value),
                    Permissions = jwtToken.Claims.Where(x => x.Type == "Permissions").Select(y => y.Value).ToList()
                };
            }
            catch (Exception e)
            {
                _loggerService.Error($"DecryptToken Exception: {e}");
                return null;
            }
        }
    }
}