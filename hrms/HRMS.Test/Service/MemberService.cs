using ExpectedObjects;
using HRMS.Enum;
using HRMS.Exceptions;
using HRMS.Models.Requests;
using HRMS.Models.Responses;
using HRMS.Repositories.Interfaces;
using HRMS.Services;
using HRMS.Services.Interfaces;
using Microsoft.AspNetCore.Http;
using NSubstitute;
using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Globalization;

namespace HRMS.Test.Service
{
    public class Tests
    {
        private ISha512Service _sha512Service;
        private IMemberRepository _memberRepository;
        private IMemberService _memberService;
        private IWorkingDateService _workingDateService;
        private IMemberDataService _memberDataService;

        // [SetUp]
        // public void Setup()
        // {
        //     _sha512Service = Substitute.For<ISha512Service>();
        //     _memberRepository = Substitute.For<IMemberRepository>();
        //     _workingDateService = Substitute.For<IWorkingDateService>();
        //     _memberDataService = Substitute.For<IMemberDataService>();
        //     _memberService = new MemberService(_sha512Service, _memberRepository,
        //         _workingDateService, _memberDataService);
        //
        //     _memberDataService.GetCurrentMemberId().Returns(1);
        // }

        [Test]
        public void DoClock_should_throw_exception_when_timezone_is_not_provided()
        {
            var doClockRequest = new DoClockRequest();

            Assert.That(() => _memberService.DoClock(doClockRequest), 
                Throws.Exception.TypeOf<ApiException>()
                .With.Property("Error").EqualTo(ApiErrorEnum.InvalidModelState)
                .With.Message.EqualTo("TimeZone is required"));
        }

        [Test]
        public void DoClock_should_throw_exception_when_user_is_not_at_company_and_location_is_not_provided()
        {
            var doClockRequest = new DoClockRequest()
            {
                TimeZone = DateTimeOffset.Now.Offset
            };

            Assert.That(() => _memberService.DoClock(doClockRequest), 
                Throws.Exception.TypeOf<ApiException>()
                .With.Property("Error").EqualTo(ApiErrorEnum.InvalidModelState)
                .With.Message.EqualTo("Location is required when not within company"));
        }

        [Test]
        public void DoClock_should_throw_exception_when_user_has_leave_request_on_that_date()
        {
            var doClockRequest = new DoClockRequest()
            {
                TimeZone = DateTimeOffset.Now.Offset,
                IsInCompany = true,
            };

            _memberRepository.DoClock(doClockRequest).Returns(new RepositoryBaseResponse
            {
                ErrorCode = ApiErrorEnum.ClockFailedMemberOnLeave,
            });

            Assert.That(() => _memberService.DoClock(doClockRequest),
                Throws.Exception.TypeOf<ApiException>()
                .With.Property("Error").EqualTo(ApiErrorEnum.ClockFailedMemberOnLeave));
        }

        [Test]
        public void DoClock_should_throw_exception_when_user_has_requested_to_reclock_on_that_date()
        {
            var doClockRequest = new DoClockRequest()
            {
                TimeZone = DateTimeOffset.Now.Offset,
                IsInCompany = true,
            };

            _memberRepository.DoClock(doClockRequest).Returns(new RepositoryBaseResponse
            {
                ErrorCode = ApiErrorEnum.AlreadyReClock,
            });

            Assert.That(() => _memberService.DoClock(doClockRequest),
                Throws.Exception.TypeOf<ApiException>()
                .With.Property("Error").EqualTo(ApiErrorEnum.AlreadyReClock));
        }

        [Test]
        public void CheckClockStatus_should_throw_exception_when_date_is_not_provided()
        {
            var clockStatusRequest = new ClockStatusRequest();

            Assert.That(() => _memberService.CheckClockStatus(clockStatusRequest),
                Throws.Exception.TypeOf<ApiException>()
                .With.Property("Error").EqualTo(ApiErrorEnum.InvalidModelState)
                .With.Message.EqualTo("Date is required"));
        }

        [Test]
        public void DoReClock_should_throw_exception_when_date_is_not_provided()
        {
            var reclockRequest = new DoReClockRequest();

            Assert.That(() => _memberService.DoReclock(reclockRequest),
                Throws.Exception.TypeOf<ApiException>()
                .With.Property("Error").EqualTo(ApiErrorEnum.InvalidModelState)
                .With.Message.EqualTo("Date is required"));
        }

        [Test]
        public void DoReClock_should_throw_exception_when_time_is_not_provided()
        {
            var reclockRequest = new DoReClockRequest
            {
                Date = DateTimeOffset.Now
            };

            Assert.That(() => _memberService.DoReclock(reclockRequest),
                Throws.Exception.TypeOf<ApiException>()
                .With.Property("Error").EqualTo(ApiErrorEnum.InvalidModelState)
                .With.Message.EqualTo("Time is required"));
        }

        [Test]
        public void DoReClock_should_throw_exception_when_location_is_not_provided()
        {
            var reclockRequest = new DoReClockRequest
            {
                Date = DateTimeOffset.Now,
                Time = DateTimeOffset.Now.Offset
            };

            Assert.That(() => _memberService.DoReclock(reclockRequest),
                Throws.Exception.TypeOf<ApiException>()
                .With.Property("Error").EqualTo(ApiErrorEnum.InvalidModelState)
                .With.Message.EqualTo("Location is required"));
        }

        [Test]
        public void DoReClock_should_throw_exception_when_reason_is_not_provided()
        {
            var reclockRequest = new DoReClockRequest
            {
                Date = DateTimeOffset.Now,
                Time = DateTimeOffset.Now.Offset,
                Location = "Company"
            };

            Assert.That(() => _memberService.DoReclock(reclockRequest),
                Throws.Exception.TypeOf<ApiException>()
                .With.Property("Error").EqualTo(ApiErrorEnum.InvalidModelState)
                .With.Message.EqualTo("Reason is required"));
        }

        [Test]
        public void DoReClock_should_throw_exception_when_location_length_is_longer_than_200()
        {
            var reclockRequest = new DoReClockRequest
            {
                Date = DateTimeOffset.Now,
                Time = DateTimeOffset.Now.Offset,
                Location = new string('a', 201),
                Reason = "Forgot"
            };

            Assert.That(() => _memberService.DoReclock(reclockRequest),
                Throws.Exception.TypeOf<ApiException>()
                .With.Property("Error").EqualTo(ApiErrorEnum.InvalidModelState)
                .With.Message.EqualTo("Maximum characters allowed for location is 200"));
        }

        [Test]
        public void DoReClock_should_throw_exception_when_reason_length_is_longer_than_200()
        {
            var reclockRequest = new DoReClockRequest
            {
                Date = DateTimeOffset.Now,
                Time = DateTimeOffset.Now.Offset,
                Location = "Company",
                Reason = new string('a', 201)
            };

            Assert.That(() => _memberService.DoReclock(reclockRequest),
                Throws.Exception.TypeOf<ApiException>()
                .With.Property("Error").EqualTo(ApiErrorEnum.InvalidModelState)
                .With.Message.EqualTo("Maximum characters allowed for reason is 200"));
        }

    }
}