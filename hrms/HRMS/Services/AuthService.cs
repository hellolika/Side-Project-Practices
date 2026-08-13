using HRMS.Models;
using HRMS.Models.Requests;
using HRMS.Models.Responses;
using HRMS.Repositories.Interfaces;
using HRMS.Services.Interfaces;

namespace HRMS.Services
{
    public class AuthService : IAuthService
    {
        private readonly ISha512Service _sha512Service;
        private readonly IMemberRepository _memberRepository;
        private readonly IBackOfficeRepository _backOfficeRepository;
        private readonly IJwtService _jwtService;

        public AuthService(ISha512Service sha512Service, IMemberRepository memberRepository, IJwtService jwtService,
            IBackOfficeRepository backOfficeRepository)
        {
            _sha512Service = sha512Service;
            _memberRepository = memberRepository;
            _jwtService = jwtService;
            _backOfficeRepository = backOfficeRepository;
        }

        public ApiBaseResponse<LoginResponse> Login(LoginRequest request)
        {
            request.CheckModels();
            var dbPassword = _sha512Service.Encrypt(request.Password);
            var dbLoginResponse = _memberRepository.Login(request.Email, dbPassword);
            dbLoginResponse.CheckErrorCode();
            var memberPermission = _memberRepository.GetPermissionsByMemberId(dbLoginResponse.MemberId);
            dbLoginResponse.Permissions = memberPermission
                .Select(i => new GetAllPermissionResponse {Id = i.PermissionId, PermissionName = i.PermissionName})
                .DistinctBy(p => p.Id).ToList();
            var memberRoles = _backOfficeRepository.GetRolesBy(dbLoginResponse.MemberId);
            dbLoginResponse.Role = memberRoles.DistinctBy(p => p.Id).Select(role => role.Id).ToList();
            dbLoginResponse.Jwt = _jwtService.GenerateToken(new JwtData(dbLoginResponse, memberPermission));
            // Check if the member can see the salary of other members, conditions of the member has IsCanSeeMemberSalary = true or the member is an admin
            dbLoginResponse.IsCanSeeMemberSalary =
                _memberRepository.GetProfile(dbLoginResponse.MemberId).IsCanSeeMemberSalary || dbLoginResponse.Role.Contains(1); 
            return new ApiBaseResponse<LoginResponse>(dbLoginResponse);
        }

        public bool IsMemberExist(int id)
        {
            return _memberRepository.IsMemberExist(id);
        }

        public bool TryDecryptToken(string token, out JwtData jwtDataResult)
        {
            jwtDataResult = _jwtService.DecryptToken(token);
            return jwtDataResult != null;
        }
    }
}