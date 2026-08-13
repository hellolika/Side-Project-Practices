using HRMS.Models.Requests;
using HRMS.Models.Responses;
using HRMS.Services.Interfaces;
using Microsoft.AspNetCore.Mvc;

namespace HRMS.Controllers
{
    [Route("/api/[controller]")]
    [ApiController]
    public class AuthController : Controller
    {
        private readonly IAuthService _authService;

        public AuthController(IAuthService authService)
        {
            _authService = authService;
        }

        [HttpPost("Login")]
        public ApiBaseResponse<LoginResponse> Login(LoginRequest request)
        {
            return _authService.Login(request);
        }
    }
}